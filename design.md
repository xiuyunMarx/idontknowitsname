# Design

We build our system on byLLM in Jac, the first runtime to introduce Meaning-Typed Programming (MTP) and the one with the most complete ecosystem around it. The system is a serving layer between byLLM agents and one inference engine. It has three parts. A decompiler maps every incoming request back to the program construct that produced it. A workflow model, one per agent program, is learned online from the decompiled traces; it consists of an annotated probabilistic finite automaton, a next-call predictor, and a set of data-flow rules. A planner turns the model's predictions into KV-cache work with deadlines and pushes the resulting priorities into the engine's two-tier cache. Every served request updates the model, and every update re-plans the sessions in flight.

## Recover semantic execution state

### Recover program constructs

The byLLM runtime lowers each construct into a request by a fixed template. A `by llm` call is rendered as a system prompt, the function's qualified name and typed signature, the `sem` description of each parameter and of the return type, a schema block, the argument values, an optional view of the receiving object, and a trailing schema hint. A `visit ... by llm` routing decision is rendered as a fixed routing instruction, views of the walker and the current node, and a list of candidate nodes identified by handles. In both templates only the argument values, the object views, and the candidate list depend on the execution; everything else is determined by the program text.

The decompiler inverts the lowering along this split. The structural part is hashed into a call-site identity, so every request emitted by the same construct, in any session, maps to the same symbol. The value part is kept as an observation attached to that call: argument bindings, receiver state, offered candidates, and the model's reply. A typed retry or the next ReAct turn re-sends the transcript of the original call extended by feedback or tool results; the decompiler recognises the extension and treats it as a continuation of the same logical call rather than a new one. Without this rule a tool-using call would appear to the workflow model as a loop of unbounded length. The output of this stage is a session trace: a sequence of call-site symbols, each carrying its observation and its arrival and completion times.

### Recover state transitions and data dependency

#### Recover data dependency

For every parameter of every call site the model learns where the value comes from within a session. Each observed value is explained against the values earlier calls in the same session consumed or produced, using a fixed hierarchy of provenance forms from most to least specific:

| form | meaning |
|---|---|
| copy | the same parameter carried the identical value earlier in the session (a session constant) |
| extend | the same parameter carried a value that is a proper prefix of this one (an accumulating history) |
| response | the value is an earlier call's reply, or a field of it |
| take | the value is another parameter of an earlier call |
| candidate | the value is the node an earlier routing decision selected |
| constant | no in-session explanation; the literal is recorded and gains support only if it repeats across sessions |

Counts of these explanations accumulate per parameter across sessions. A rule is adopted when one form has at least two observations and more than all others combined, and it is trusted for speculation only after prompts regenerated under it have matched real prompts byte for byte.

The rules assign every part of a prompt a lifetime: constant across sessions, constant within a session, growing within a session, or fresh at every call. Once every parameter of a site has a settled rule, the site's binding layout is frozen: parameters are ordered by decreasing lifetime, and subsequent requests are re-emitted in that order before they reach the engine. The reorder breaks the site's cache prefix exactly once. After it, the r-th invocation of a site is a byte-level extension of the (r-1)-th: header, session constants, and history so far form a shared prefix, while the fresh bindings and the reply form the tail. The rules also define a reconstruction function that, given a site and the live session state, renders the longest prefix of the site's next prompt that is already determined. This function is what lets the planner name the exact reusable bytes of a future call rather than merely the agent that will send it.

#### Recover state transition

We treat an agent as an annotated probabilistic finite automaton. Its alphabet is the set of call sites. A state is a call site in an execution context, each transition carries the probability of the next call site, and each state carries a termination probability. Two timing annotations are attached from history: every state holds the distribution of its call's duration, and every transition holds the distribution of the gap between the source call's completion and the target call's arrival. Transitions leaving a routing decision additionally record how often the branch was taken against how often it was offered, giving the branch probability conditioned on availability.

**Learning by state merging.** Completed sessions are arranged into a prefix tree in which every path is an observed sequence and every node counts the sessions that reached it and the sessions that ended there. The tree is reduced in the red-blue order of Alergia: a blue node is merged into a red node when the two are compatible, and otherwise promoted to red. Two nodes are compatible when their termination rates, their rates for every outgoing symbol, and recursively their successors satisfy the Hoeffding test

$$\left|\frac{c_1}{n_1}-\frac{c_2}{n_2}\right| < \sqrt{\tfrac{1}{2}\ln\tfrac{2}{\alpha}}\left(\frac{1}{\sqrt{n_1}}+\frac{1}{\sqrt{n_2}}\right), \qquad \alpha = 0.05 .$$

Merging is restricted to nodes of the same call site, so a state never conflates two constructs, and two occurrences of one site are kept apart only when their observed futures differ. The automaton is rebuilt when the session count reaches 8, 16, 32 and so on. Because a rebuild creates fresh states with no timing, every recorded session is replayed onto the new automaton, so state-specific gap and duration quantiles survive the rebuild instead of collapsing to global averages.

**Next-call prediction.** Merged transition probabilities converge slowly when sessions are few, so the next-call distribution comes from a companion estimator: a context trie over the last k = 4 call sites, smoothed with Witten-Bell backoff. Estimates from the empty context up to the full 4-symbol context are blended shallow to deep, each context weighted by n/(n+T) where n is its sample size and T the number of distinct outcomes seen there. A deep context seen once therefore shades, but does not override, its well-supported shorter suffixes. The same estimator gives the probability that the session ends.

Prediction expands a tree of continuations rather than a single most likely chain: from the current session state the top 3 next calls are expanded at every step, to depth 8, pruning paths below a probability floor or beyond a time horizon. For every future call site the probability mass of all paths reaching it is summed. Arrival times come from the automaton: along a path the median arrival t50 is the sum of the median gaps and durations of the hops, and the spread compounds under an independent-hop approximation as the root of the summed squared quantile gaps, giving t90. Each predicted call thus carries a probability p and arrival quantiles t50 and t90, measured from the completion of the last call.

## Proactive scheduling during runtime

With the automaton and the data-flow rules, the planner holds for every live session a forecast of upcoming work: which call sites will fire, with what probability, in what window, and with which prefix already determined. After every completed call the session is re-planned from scratch, since its branches may have collapsed.

**Jobs.** Each predicted call becomes a job carrying its reconstructed prefix, split into the tokens that are uncached anywhere and the tokens resident on the host tier. Its deadline is the predicted arrival tightened by the forecast's spread,

$$\text{deadline} = \text{anchor} + t_{50} - 0.5\,(t_{90}-t_{50}),$$

where the anchor is the completion time of the last call, and its value is the expected work it saves,

$$\text{value} = p \cdot (\text{uncached} + 0.8 \cdot \text{host}),$$

host reloads being cheaper than recomputation. Jobs run earliest deadline first. A job is released only when the deadline minus its remaining work, costed at the engine's profiled prefill rate and host-to-device rate, has arrived, and only if it can finish before another session's next call is expected, so speculative work never delays a request being served.

**Promotion.** The main job kind reloads a predicted call's host-resident prefix onto the device before the call arrives. A promotion may use free device slots and the cache of sessions that have ended, never active cache, and it must leave one full prefill chunk free. When there is no room the job is deferred and all promotions sit out a cooldown, rather than displacing cache that a live session may need.

**Creation.** Speculative prefill of an entire predicted prompt is disabled in the managed configuration used for the experiments; promotion and eviction steering carry the managed arm.

**Eviction steering.** The plan is mirrored into the engine's eviction order. Every node of the device radix tree is placed in a band, lower bands evicted first and LRU inside a band:

| band | contents |
|---|---|
| retired | private bytes of sessions that have ended |
| transient | the one-off tail of a served prompt, past the head the data-flow rules can rebuild |
| unclassified | everything else, left to recency |
| planned | prefixes some job needs, ordered by reuse score |

The reuse score of a planned prefix is

$$s = p \cdot \exp(-\Delta t / 60\,\text{s}),$$

the probability of the call decayed by its predicted time to arrival. The engine sums the scores of every plan covering a node, so a header shared by many sessions outranks any single session's private history, and among private prefixes the sooner and surer use is kept longest. After a call is served, the part of its prompt beyond the reconstructible head is demoted to the transient band, since no future call will extend it.

The host tier is deliberately not steered. It evicts retired cache first and everything else by recency, so a private prefix the device gives up stays reloadable until recency retires it. Applying the plan's bands to both tiers would purge from the host exactly the prefixes the device sacrifices under overload, turning a single-tier miss into a double-tier miss; leaving the host to recency keeps the plan's mistakes recoverable.

The scheduler is aggressive in what it prepares and conservative in what it sacrifices. It can afford to be aggressive because the workflow model names not just the next agent but the exact bytes it will send and approximately when, and it must be conservative because under load a forecast that displaces present work for speculative work loses more than it gains.
