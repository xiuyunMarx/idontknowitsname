# Storyline

Efficient serving for LLM agents depends on knowing more about a request than its text. A serving system that knows which callsite a request comes from, which parts of its prompt will recur, and how long each part stays useful can lay out prompts for reuse, keep the right KV-cache entries resident, and prepare for calls that have not arrived yet. Recent systems obtain such knowledge in one of two ways. Explicit approaches ask the application to annotate it: Parrot exposes semantic variables, Prompt Cache requires a prompt schema, and Pythia derives workflow structure from developer-provided traces. Implicit approaches infer it from the request stream alone: CacheScout learns transitions between prompt-prefix fingerprints online, and PBKV predicts future invocations from workflow history and learned representations of request content. Explicit approaches see program structure but change the application interface; implicit approaches keep the interface but see only opaque text, so the structure they recover stops at "this prefix was seen before". (Parrot, Prompt Cache, Pythia, CacheScout, PBKV)

Language-integrated LLM programming systems remove this trade-off. In these systems, requests are not authored as free-form text. A compiler or language runtime lowers program constructs into model requests through a fixed procedure: MTP/byLLM synthesizes prompts from semantic function signatures, parameter and return types, object schemas, semantic annotations, and dynamically bound argument values; NVIDIA NOOA derives agent prompts and interfaces from Python objects, methods, fields, docstrings, and type annotations. The request that reaches the serving system is therefore a serialized artifact of a deterministic lowering process, and the program structure survives in it: the instruction text identifies the callsite, the layout identifies each bound parameter, and the values reveal where each parameter came from. Existing serving systems discard all of this and process the request as undifferentiated text. (MTP/byLLM, NOOA)

Our key insight is that these generated request formats constitute an implicit compiler-serving ABI. Because the lowering is systematic, the serving layer can invert it: with knowledge of how a runtime lowers program constructs, ordinary requests can be decompiled back into a callsite-level semantic IR without any annotation from the application. The IR carries three pieces of information, and each one unlocks a KV-cache optimization that is unavailable to a serving system that only sees text.

1. **Callsite identity.** Every request is resolved to the language construct that produced it at admission time, before prefill. Sessions become sequences of callsite symbols with timing, from which the server learns a probabilistic program model online and forecasts upcoming calls early enough to promote their prefixes from host to device before they arrive.
2. **Binding boundaries.** Each request decomposes into a static callsite header and a set of parameter bindings. Because the boundaries are known, the server can reorder bindings by stability, moving cross-session constants and session-stable values ahead of volatile ones while keeping every block byte-for-byte intact, so that repeated invocations share the longest possible token prefix.
3. **Value provenance.** By tracing each binding's value across calls in a session, the server classifies it as a cross-session constant, a copy of an earlier value, an extension of an earlier value, or a one-off. This tells the cache which parts of a prompt will be reconstructed by later calls and which parts are private tails that will never be seen again, and which prefixes belong to sessions that have finished, so eviction can be ordered by expected lifetime instead of recency.

We build these three optimizations on a single serving layer between byLLM agents and an SGLang engine. Our approach turns structure that compiler-generated requests already contain into actionable knowledge for the serving system, recovering the benefits of explicit program-aware serving without changing the application programming interface.

# Design
We build our system on byLLM in Jac, the first runtime to introduce Meaning-Typed Programming (MTP).  The system is a serving layer between byLLM agents and one inference engine, based on Sglang. The system comprises 3 layers:
1. A decompiler that invert the request back into original program constructs.
2. A workflow model which maintains the probalistic finite automaton of program and updates online. 
3. A planner turns the model's predictions into KV-Cache scheduler

## Recover program constructs
ByLLM runtime lowers program constructs into model in a specific way. Specifically, the prompt can be structurally divided into 2 parts: Invariant structural part and the value part. Invariant part is the instruction on the task, and the value part is the specific context. The output of this stage is a session trace: a sequence of call-site symbols, each carrying its observation and its arrival and completion times. From the observed trace sequence, more detailed execution patterns can be extracted.

## Recover state transition
Our Design treats an agent as a timing-annotated probabilistic finite-state model:

$$\mathcal{A} = (Q, P, q_0, F, \mathcal{T})$$
where:
1. $Q$ is the set of states, with each state $(q \in Q$) representing an LLM callsite.
2. $P$ is the transition probability for each state. 
3. $q_0$ is the entrance callsite.
4. $F$ is probablity of termination for an agent in each state.
5. $\mathcal{T}=\left\{D^{\mathrm{exec}}_q,D^{\mathrm{gap}}_{q,q'}\right\}$ is the timing annotations associated with states and transitions. Where $D^{\mathrm{exec}}_q$ is the execution-time distribution of the LLM callsite represented by state $q$, and $D^{\mathrm{gap}}_{q,q'}$ is the inter-call gap distribution between the completion of callsite $q$ and the invocation of its successor $q'$

In one session, server can only observe a chain of LLM calls. For each session, the chain is firstly built into a prefix tree with probalistic transition, then doing Alergia-style state merge each time when the session count reaches $8 \times 2^k$. Two states are merged when their behaivors are statistically indistinguishable, determined by the following principle:

$$\left|\frac{c_1}{n_1}-\frac{c_2}{n_2}\right| < \sqrt{\tfrac{1}{2}\ln\tfrac{2}{\alpha}}\left(\frac{1}{\sqrt{n_1}}+\frac{1}{\sqrt{n_2}}\right), \qquad \alpha = 0.05 .$$

where $c_i$ is the transition frequency and $n_i$ is total number of recorded transitions on this state. 

## Runtime Proactive Scheduling

At runtime, the server maintains a planning state for every live session. Let

$$h_t=(x_1,\ldots,x_t)$$

denote the sequence of logical LLM callsites observed in the session. **After admitting or completing a logical call**, the server uses the learned program model to forecast future calls and translates the forecast into a KV-cache residency and eviction plan.

A forecast is represented as

$$
\widehat{\mathcal C}_t=\left\{
(x,p_x,t_{50,x},t_{90,x})
\right\},
$$

where $x$ is a future callsite, $p_x$ is its estimated reachability mass, and $t_{50,x}$ and $t_{90,x}$ are its estimated arrival-time quantiles relative to the completion of the latest observed call.


### Future Call Anticipation

Transition-frequency estimated from the merged automaton may be unstable when only a small number of sessions have been observed. We therefore use a companion bounded-context predictor for next-call probabilities.

### Context-Based Next-Call Prediction

The predictor maintains a context trie containing callsite subsequences of length at most $K=4$. Given the current history $h$, it considers suffixes ranging from the empty context to the longest observed suffix of $h$.

For a context $u$ (Context $u$ here is the recent consecutive callsites), let $C_u(x)$ be the number of times callsite $x$ followed $u$, and let $C_u(\bot)$ be the number of sessions that terminated after $u$. Define the total number of observed outcomes as:

$$n_u=C_u(\bot)+\sum_x C_u(x)$$

and the number of distinct outcomes as:
$$
T_u=
\left|\left\{x:C_u(x)>0\right\}\right|+
\mathbf{1}[C_u(\bot)>0]
$$

Then the confidence assigned to the context is

$$
\lambda_u=\frac{n_u}{n_u+T_u}.
$$

The next-call distribution is obtained using Witten-Bell-style interpolation to reduce the impact of rarely occurring long context:

$$
\widehat P(x\mid u)=
\lambda_u\frac{C_u(x)}{n_u}
+
(1-\lambda_u)
\widehat P(x\mid \operatorname{suffix}(u)),
$$

and the termination probability is computed analogously:

$$
\widehat P(\bot\mid u)=
\lambda_u\frac{C_u(\bot)}{n_u}
+
(1-\lambda_u)
\widehat P(\bot\mid \operatorname{suffix}(u)).
$$

A well-supported long context dominates the prediction, whereas a context observed only a few times merely adjusts the more stable short-context distribution. Termination probability is treated as probability mass that produces no successor in the prediction tree.

### Estimating Cumulative Reachability

Cumulative Reachability is the probablity of accessing a future calls within a time window. It is estimated by a bounded best-first expansion of future executions.

Define frontier element, which represents the current search state of a hypothetical execution path; A frontier element is written as

$$
(\pi,h,q,u_{50},s^2,d),
$$

where:

- $\pi$ is the probability of the partial execution path;
- $h$ is the hypothetical call history along that path;
- $q$ is the automaton state reached after the last call on the hypothetical path. For a non-root frontier element, that state corresponds to the anticipated callsite where the hypothetical path currently ends;
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


The search expands the highest-probability frontier first and applies the following default bounds:

$$
p_{\min}=0.02,
\qquad
H=120\text{ s},
\qquad
K_{\mathrm{branch}}=3,
$$

with a maximum depth of 8 and at most 64 expanded frontier nodes. A path is pruned if its probability falls below $p_{\min}$, its median arrival exceeds the planning horizon $H$, or it exceeds one of the structural bounds.

Multiple paths may reach the same callsite. Their  aggregated path masses, which is also the cumulative reachability for $x$, is:

$$
p_x=
\min\left(
1,\sum_{\rho:\rho\leadsto x}P(\rho)
\right).
$$


## Runtime KV-Cache Scheduling

### Prompt Re-layout for More Efficient KV Reuse

KV-cache reuse requires two prompts to share an identical token prefix. In the original request layout, a volatile parameter may appear before parameters that remain stable across calls, causing the reusable prefix to terminate early. The server therefore learns a stability-aware binding order for every LLM callsite and applies it to subsequent requests.

For every binding $b$ of callsite $x$, the server infers a provenance rule from completed session traces. A binding is classified as a cross-session constant when the same literal repeatedly appears across sessions, as a session-stable copy when its value is reused from an earlier call in the same session, or as an extension when its new value grows from an earlier value, such as an accumulating interaction history. Bindings derived from per-call values or without a stable provenance are treated as volatile.

A provenance rule $r$ for binding $b$ is accepted once it accouts for at least half of all observations for that binding, and has at least two supporting observations
The most frequently observed rule

The server assigns each binding a stability rank:

$$
\operatorname{rank}(b)=
\begin{cases}
0, & b\text{ is constant across sessions},\\
1, & b\text{ is copied within the session},\\
2, & b\text{ extends an earlier value},\\
3, & \text{otherwise}.
\end{cases}
$$

After every binding of a callsite has been assigned with a provenance rule $r$, the server performs a stable sort using this ranking and permanently freezes the resulting layout. 

For subsequent requests to the callsite, the server moves stable and accumulating binding blocks before volatile ones while preserving every block byte-for-byte. 

The resulting prompt has the logical layout

$$
\underbrace{\text{static callsite header}}_{\text{shared across calls}}
\;\Vert\;
\underbrace{\text{cross-session constants}}_{\text{longest lifetime}}
\;\Vert\;
\underbrace{\text{session-stable values}}_{\text{shared within a session}}
\;\Vert\;
\underbrace{\text{accumulating values}}_{\text{growing prefix}}
\;\Vert\;
\underbrace{\text{volatile values}}_{\text{private tail}}.
$$

This organization maximizes the longest common prefix between repeated invocations. 


### Deadline-Driven Promotion
#### Estimating Job Deadline
For a predicted call

$$
c=(x,p,t_{50},t_{90}),
$$

let $a$ be the session's current timing anchor. Its scheduling deadline is

$$
d_c=
\max\left(
t_{\mathrm{now}},
a+t_{50}
-\kappa(t_{90}-t_{50})
\right),
$$

where $\kappa=0.5$. The term $\kappa(t_{90}-t_{50})$ is to moves uncertain jobs earlier and let scheduler to prepare the prefix more conservatively.

The absolute pessimistic arrival time

$$
\bar t_c=
\max(d_c,a+t_{90})
$$

is  for garbage collection. If the predicted call has not arrived after $\bar t_c$, its cache-plan entry is considered stale and discarded.

#### Deadline-Driven KV Promotion

Let $H_c$ be the number of target-prefix tokens available in host memory but absent from device memory, and let $R_{\mathrm{promote}}$ be the profiled host-to-device promotion rate. The latest safe promotion time is approximated by

$$
r_c=
d_c
-\frac{H_c}{R_{\mathrm{promote}}}.
$$

A promotion is triggered when the engine is idle or when the current time approaches $r_c$. Among eligible promotions, the scheduler prioritizes the earliest deadline.

Promotion can consume free device capacity or reclaim cache belonging to retired sessions, but it does not evict cache associated with active sessions. 



### Prediction-Aware Eviction

For a predicted call $c$, its time-discounted reuse score is

$$
S_c(t)=
p_c
\exp\left(
-\frac{\max(0,d_c-t)}{\tau}
\right),
\qquad
\tau=60\text{ s}.
$$

Thus, a prefix needed sooner receives greater protection than an equally probable prefix needed much later.

Prompts are stored in a radix tree, a radix node may be referenced by multiple sessions. For a radix node $v$, the protection score is aggregated across live sessions:

$$
S(v,t)=
\sum_s
\max_{\substack{c\in\mathcal J_s\\v\preceq c}}
S_c(t),
$$

where $\mathcal J_s$ is the set of predicted calls for session $s$, and $v \preceq c$ means callsite $c$ will use node $v$


Device-resident KV entries are divided into lifecycle-aware priority bands, ordered by ascending importance:

1. **retired** -- private cache belonging to completed sessions;
2. **transient** -- one-off prompt tails that value-flow analysis cannot reconstruct or reuse;
3. **unclassified** -- ordinary cache not covered by the current plan;
4. **planned** -- prefixes covered by the active cache plan.

KV entries within the planned band are ordered using their aggregated reuse scores, while KV Entries within the other bands retain recency-based ordering.

After a request completes, the reusable reconstructed head of its prompt remains normally cacheable, while the private, non-reconstructible tail is demoted to the transient band. When the session terminates, its private prefixes enter the retired band and become the first source of reclaimable capacity.

The host cache policy remains LRU. This preserves prefixes evicted from the device as recoverable promotion candidates. Applying the same prediction policy to both tiers could remove a mispredicted prefix from the device and host simultaneously, turning a recoverable device miss into full recomputation.



# Experiments
## Workload

A HoVer-style multi-hop fact checker, using qwen 8b on 3090. Using SGLang kernel copy backend for HiCache, host tier 16 GB. And SGLang as baseline.



## 4. Results

### 4.1 Hit rate and TTFT

| c | sessions | miss % LRU | miss % ours | host hit % LRU | host hit % ours | TTFT p50 LRU | TTFT p50 ours | TTFT p95 LRU | TTFT p95 ours |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 48 | 45.7 | 46.2 | 35.0 | 34.5 | 0.75 s | 1.00 s | 3.9 s | 4.3 s |
| 10 | 60 | 58.2 | 53.3 | 27.8 | 31.6 | 2.79 s | 2.42 s | 7.9 s | 8.2 s |
| 12 | 72 | 66.7 | 58.1 | 20.5 | 28.1 | 5.06 s | 4.50 s | 11.9 s | 10.5 s |
| 14 | 84 | 73.6 | 61.2 | 15.7 | 25.6 | 8.52 s | 6.49 s | 15.5 s | 11.5 s |
| 16 | 96 | 74.8 | 64.2 | 14.5 | 24.6 | 10.68 s | 9.80 s | 18.1 s | 16.1 s |
| 18 | 108 | 74.1 | 63.8 | 14.7 | 23.8 | 12.87 s | 10.95 s | 19.7 s | 17.6 s |


### 4.2 JCT

| c | JCT mean LRU | JCT mean ours | change | JCT p95 LRU | JCT p95 ours | steady-state change |
|---|---|---|---|---|---|---|
| 8 | 101.6 s | 108.0 s | +6 % | 184.7 s | 179.3 s | +4 % |
| 10 | 141.6 s | 137.2 s | -3 % | 251.6 s | 239.2 s | -5 % |
| 12 | 185.8 s | 170.1 s | -8 % | 323.7 s | 304.5 s | -10 % |
| 14 | 228.4 s | 185.3 s | -19 % | 405.6 s | 317.2 s | -21 % |
| 16 | 257.4 s | 230.5 s | -10 % | 491.4 s | 434.5 s | -8 % |
| 18 | 283.3 s | 253.7 s | -10 % | 516.0 s | 431.1 s | -14 % |






