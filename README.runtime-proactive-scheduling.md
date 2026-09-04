# Runtime Proactive Scheduling

At runtime, the server maintains a planning state for every live session. Let

$$
h_t=(x_1,\ldots,x_t)
$$

denote the sequence of logical LLM callsites observed in the session, where multiple HTTP turns belonging to the same ReAct or tool-using call are folded into one callsite. After admitting or completing a logical call, the server uses the learned program model to forecast future calls and translates the forecast into a KV-cache residency and eviction plan.

A forecast is represented as

$$
\widehat{\mathcal C}_t
=
\left\{
(x,p_x,t_{50,x},t_{90,x})
\right\},
$$

where $x$ is a future callsite, $p_x$ is its estimated reachability mass, and $t_{50,x}$ and $t_{90,x}$ are its estimated arrival-time quantiles relative to the completion of the latest observed call.

Planning is advisory: it only changes cache placement and eviction priorities. It never changes the agent's control flow or the result returned to the agent. Consequently, an incorrect prediction may retain or promote an unneeded prefix, but cannot alter program semantics.

## Future Call Anticipation

Although the probabilistic automaton records execution states and timing annotations, transition-frequency estimates from the merged automaton may be unstable when only a small number of sessions have been observed. We therefore use a companion bounded-context predictor for next-call probabilities and use the automaton primarily to recover context-sensitive timing.

### Context-Based Next-Call Prediction

The predictor maintains a context trie containing callsite subsequences of length at most $K=4$. Given the current history $h$, it considers suffixes ranging from the empty context to the longest observed suffix of $h$.

For a context $u$, let $C_u(x)$ be the number of times callsite $x$ followed $u$, and let $C_u(\bot)$ be the number of sessions that terminated after $u$. Define

$$
n_u
=
C_u(\bot)+\sum_x C_u(x)
$$

as the total number of observed outcomes, and

$$
T_u
=
\left|\left\{x:C_u(x)>0\right\}\right|
+
\mathbf{1}[C_u(\bot)>0]
$$

as the number of distinct outcomes. The confidence assigned to the context is

$$
\lambda_u=\frac{n_u}{n_u+T_u}.
$$

The next-call distribution is obtained using Witten-Bell-style interpolation:

$$
\widehat P(x\mid u)
=
\lambda_u\frac{C_u(x)}{n_u}
+
(1-\lambda_u)
\widehat P(x\mid \operatorname{suffix}(u)),
$$

and the termination probability is computed analogously:

$$
\widehat P(\bot\mid u)
=
\lambda_u\frac{C_u(\bot)}{n_u}
+
(1-\lambda_u)
\widehat P(\bot\mid \operatorname{suffix}(u)).
$$

Thus, a well-supported long context dominates the prediction, whereas a context observed only a few times merely adjusts the more stable short-context distribution. Termination probability is treated as probability mass that produces no successor in the prediction tree.

### Probabilistic Tree Expansion

Predicting only the most likely successor is insufficient because preparing KV cache for several plausible branches can be beneficial. The server therefore performs a bounded best-first expansion of future executions.

Each frontier element is represented as

$$
(\pi,h,q,u_{50},s^2,d),
$$

where:

- $\pi$ is the probability of the partial execution path;
- $h$ is the hypothetical call history along that path;
- $q$ is the corresponding state in the learned automaton;
- $u_{50}$ is the estimated median completion time of the previous call;
- $s^2$ is the accumulated squared timing spread;
- $d$ is the prediction depth.

The initial frontier contains

$$
(1,h_t,q_t,0,0,0).
$$

For a candidate successor callsite $x$, the path probability becomes

$$
\pi'=\pi\widehat P(x\mid h).
$$

Let $q'$ be the automaton state reached by $x$, and let

$$
g_{50}=Q_{0.5}\left(D^{\mathrm{gap}}_{q,q'}\right),
\qquad
g_{90}=Q_{0.9}\left(D^{\mathrm{gap}}_{q,q'}\right).
$$

The predicted arrival time of $x$ is

$$
a_{50}=u_{50}+g_{50},
$$

with accumulated spread

$$
s_a^2=s^2+(g_{90}-g_{50})^2,
$$

and pessimistic arrival estimate

$$
a_{90}=a_{50}+\sqrt{s_a^2}.
$$

To expand beyond $x$, the execution time of $x$ is added. Let

$$
e_{50}=Q_{0.5}\left(D^{\mathrm{exec}}_{q'}\right),
\qquad
e_{90}=Q_{0.9}\left(D^{\mathrm{exec}}_{q'}\right).
$$

The successor frontier state is then

$$
\left(
\pi',
h\mathbin{\|}x,
q',
a_{50}+e_{50},
s_a^2+(e_{90}-e_{50})^2,
d+1
\right).
$$

This root-sum-square construction approximates independent timing variation across inter-call gaps and call executions. It is used as a scheduling heuristic rather than as an exact composition of statistical quantiles.

The search expands the highest-probability frontier first and applies the following default bounds:

$$
p_{\min}=0.02,
\qquad
H=120\text{ s},
\qquad
K_{\mathrm{branch}}=3,
$$

with a maximum depth of eight and at most 64 expanded frontier nodes. A path is pruned if its probability falls below $p_{\min}$, its median arrival exceeds the planning horizon $H$, or it exceeds one of the structural bounds.

Multiple paths may reach the same callsite. Their path masses are aggregated as

$$
p_x
=
\min\left(
1,\sum_{\rho:\rho\leadsto x}P(\rho)
\right).
$$

For scheduling conservatism, the timing annotations of the earliest-arriving path are retained. Therefore, $p_x$ should be interpreted as bounded cumulative reachability mass rather than a strict "occurs at least once" probability in workflows containing loops.

### Online Plan Refresh

The forecast is refreshed at two runtime events.

When a new logical call arrives, its callsite is appended to the session history and the session epoch is incremented. Cache plans from older epochs are invalidated immediately. The server then predicts the successors of the newly arrived call while that call is still being decoded. Since its completion time is not yet known, the request arrival time is temporarily used as the timing anchor. This intentionally produces conservative, earlier deadlines and allows cache promotion to overlap the current decode.

When the call completes, the server replaces the anchor with the actual completion time and recomputes the plan. At this point, the model reply and additional value-flow observations may also identify a larger reusable prefix for future calls. The refreshed plan replaces the previous cache-plan set; entries associated with branches that are no longer predicted are invalidated, while unchanged entries in the same epoch retain their current cache-residency state.

This replace-on-refresh policy ensures that collapsed branches promptly lose cache protection and stop influencing promotion decisions.

## KV-Cache Scheduling

Each anticipated call is converted into a KV-cache plan entry. An entry contains the target token prefix, session and epoch identifiers, predicted probability, arrival window, and current residency across the device and host cache tiers.

### Future-Prefix Identification

Before creating a job, the server attempts to reconstruct the predicted call's user message from the live session state. Learned value-flow rules identify prompt fields that are:

- constant across sessions;
- constant within a session;
- copied from an earlier call;
- extended from an accumulating history;
- extracted from an earlier response;
- selected by an earlier routing decision;
- or freshly generated and therefore not yet predictable.

If every field is resolved and the reconstruction logic has previously reproduced a real request byte-for-byte, the complete future prompt is used as the target cache key. Otherwise, the server identifies the longest deterministic prefix currently available. This prefix may contain the static callsite header, session constants, and the known portion of an accumulating history. If dynamic reconstruction does not extend the prefix, the static callsite prefix is used as a fallback.

Consequently, a prediction can remain useful even when the exact future request is unknown: the scheduler can protect or promote the portion already cached and guaranteed to prefix the eventual request. Tokens absent from both cache tiers are not computed proactively; they are processed only when the real request arrives.

### Cache Actions

The cache plan drives two actions.

A **protection action** raises the eviction priority of a predicted prefix that is already resident in device memory. The action performs no model computation and creates no new KV entries; it only reduces the likelihood that reusable cache is evicted before the request arrives.

A **promotion action** reloads a predicted prefix already present in the host cache into device memory. It is considered only when the prefix can be recovered from the host tier without model computation.

If the prefix is absent from both tiers, the scheduler records no proactive cache action. The ordinary request path computes and inserts the missing KV state when the actual call is served.

### Deadline Assignment

For a predicted call

$$
c=(x,p,t_{50},t_{90}),
$$

let $a$ be the session's current timing anchor. Its scheduling deadline is

$$
d_c
=
\max\left(
t_{\mathrm{now}},
a+t_{50}
-\kappa(t_{90}-t_{50})
\right),
$$

where $\kappa=0.5$.

The term

$$
\kappa(t_{90}-t_{50})
$$

moves uncertain jobs earlier: a wider arrival window causes the scheduler to prepare the prefix more conservatively. The absolute pessimistic arrival time

$$
\bar t_c
=
\max(d_c,a+t_{90})
$$

is retained for garbage collection. If the predicted call has not arrived shortly after $\bar t_c$, its cache-plan entry is considered stale and discarded.

### Deadline-Driven Promotion

All host-to-device cache movements pass through a single promotion scheduler, preventing concurrent promotions from racing for device KV capacity.

Let $H_c$ be the number of target-prefix tokens available in host memory but absent from device memory, and let $R_{\mathrm{promote}}$ be the profiled host-to-device promotion rate. The latest safe promotion time is approximated by

$$
r_c
=
d_c
-\frac{H_c}{R_{\mathrm{promote}}}.
$$

A promotion becomes eligible when the engine is idle or when the current time approaches $r_c$. Among eligible promotions, the scheduler prioritizes the earliest deadline, using prediction confidence and the number of recoverable tokens as secondary preferences.

Promotion may consume free device capacity or reclaim cache belonging to retired sessions, but it does not evict cache associated with active sessions merely to satisfy a prediction. The scheduler also preserves a device-capacity reserve for incoming requests. When insufficient space is available, promotion is deferred and temporarily placed into cooldown rather than forcing an active prefix out of the cache.

### Replacement and Invalidation

Every cache-plan entry is tagged with the session epoch under which it was generated. When the session advances to another logical call, the epoch is incremented and entries from earlier epochs are marked invalid.

Within the same epoch, replanning uses replacement semantics:

- an entry appearing in both the old and new plans keeps its current residency information and receives refreshed probability and timing;
- an entry absent from the new plan is invalidated because its branch has collapsed;
- a newly predicted entry is inserted.

KV blocks associated with an invalidated entry may remain available for ordinary prefix reuse, but they no longer receive promotion or eviction priority from that session.

### Prediction-Aware Eviction

The current plan is projected into the device KV cache's eviction order. For a predicted call $c$, its time-discounted reuse score is

$$
S_c(t)
=
p_c
\exp\left(
-\frac{\max(0,d_c-t)}{\tau}
\right),
\qquad
\tau=60\text{ s}.
$$

Thus, a prefix needed soon receives greater protection than an equally probable prefix needed much later.

Because prompts are stored in a radix tree, a node may prefix several planned requests. For a radix node $v$, the protection score is aggregated across live sessions:

$$
S(v,t)
=
\sum_s
\max_{\substack{c\in\mathcal J_s\\v\preceq c}}
S_c(t),
$$

where $v\preceq c$ means that the token sequence represented by $v$ is a prefix of the cache target associated with $c$. The maximum prevents multiple predictions from one session from double-counting the same prefix, while the sum captures reuse across independent sessions. As a result, a shared system prompt needed by many sessions can receive more protection than any one session's private prefix.

Device-resident KV entries are divided into lifecycle-aware priority bands, ordered from easiest to hardest to evict:

1. **retired** -- private cache belonging to completed sessions;
2. **transient** -- one-off prompt tails that value-flow analysis cannot reconstruct or reuse;
3. **unclassified** -- ordinary cache not covered by the current plan;
4. **planned** -- prefixes covered by the active cache plan.

Entries within the planned band are ordered using their aggregated reuse scores. Entries within the other bands retain recency-based ordering.

After a real request completes, the reusable reconstructed head of its prompt remains normally cacheable, while the private, non-reconstructible tail is demoted to the transient band. When the session terminates, its private prefixes enter the retired band and become the first source of reclaimable capacity.

The host cache deliberately remains recency-managed rather than prediction-steered. This preserves prefixes evicted from the device as recoverable promotion candidates. Applying the same prediction policy to both tiers could remove a mispredicted prefix from the device and host simultaneously, turning a recoverable device miss into full recomputation.

Overall, the scheduler retains high-value device prefixes and opportunistically restores reusable host prefixes, while avoiding proactive model execution. Prediction errors therefore affect only cache efficiency and data movement; request correctness and foreground serving priority remain unchanged.
