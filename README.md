# Fuka-6.0
First Universal Kommon Ancestor - by Yasas (යසස් පොන්වීර)


Fuka-6.0 — Emergent Computation on Capacitor Substrates

Fuka-6.0 is a physics-first simulation platform exploring how computation, symbolic code, and adaptive phenotype-like behavior can emerge from a primitive network of capacitors interacting with an environment.

There are:
	•	No neurons
	•	No logic gates
	•	No symbolic rules
	•	No backpropagation
	•	No pre-programmed intelligence

Only:
	•	capacitor dynamics
	•	local plasticity
	•	energy sources
	•	environmental waves
	•	self-organization

From this, the system develops:
	•	discrete attractor states (symbols)
	•	sequences of attractors (code)
	•	evolving connection topology (hardware)
	•	adaptive long-memory pockets
	•	environment-modifying behavior (phenotype)

This is top-down and bottom-up model of how computation can arise from physical substrates.

⸻

1. Motivation

Biological computation is not designed.
It emerges from:
	1.	physical substrates
	2.	energy gradients
	3.	self-reinforcing attractors
	4.	code-like symbolic transitions
	5.	phenotype behavior that acts back on the environment

Fuka-6.0 aims to show the earliest version of this process using the simplest possible physical substrate that can compute:
a network of capacitors with leak, coupling, and local adaptation.

The goal is to study how:
	•	computation
	•	memory
	•	code
	•	hardware
	•	and phenotype

can emerge together from pure physics.

⸻

2. Capacitor Substrate Model

The substrate consists of N capacitors.
Each capacitor i has voltage V_i(t) and capacitance C_i.

Dynamics follow:

C_i \frac{dV_i}{dt}
= -\lambda_i V_i
+ \sum_j g_{ij}(V_j - V_i)
+ I_i(t)

Where:
	•	C_i — capacitance
	•	\lambda_i — leakage (natural memory decay)
	•	g_{ij} — conductance between capacitor i and j
	•	I_i(t) — environmental energy injected into capacitor i

This is the minimal physical substrate able to store and transform information.

⸻

3. Environment

The environment provides fluctuating energy input:

I_i(t) = f_i(E(t), x_i)

Where:
	•	E(t) — global environmental state
	•	x_i — spatial or structural position
	•	f_i — mapping from environment to excitation

The environment is purely physical, not symbolic.

⸻

4. Plasticity / Learning Rule

The substrate adapts using a purely local learning rule that strengthens useful connections and weakens useless ones.

\frac{dg_{ij}}{dt}
= \eta F(t)\left( V_i V_j - \alpha g_{ij} \right)

Where:
	•	\eta — learning rate
	•	\alpha — decay
	•	F(t) — global stability pressure

4.1 Stability Pressure

F(t) = -\frac{1}{N}\sum_i \left(\frac{dV_i}{dt}\right)^2

Interpretation:
	•	low turbulence → high F(t) → reinforce connections
	•	high turbulence → low F(t) → connections decay

This forms the basis of emergent “evolution.”

⸻

5. Attractors — The First Symbols

When the environment repeatedly injects energy, the substrate settles into stable states:

\mathbf{V}(t) \rightarrow A_k

Each attractor A_k is:
	•	reproducible
	•	stable under small perturbations
	•	low turbulence
	•	persistent

These attractors form the first alphabet of the system.

They are the proto-symbols.

⸻

6. Attractor Sequences — The First Code

Environmental waves arrive in discrete “slots”:
	•	energy pulse
	•	relaxation
	•	stabilize into attractor

Sampling the attractor after each slot produces:

A_{k_1}, A_{k_2}, A_{k_3}, \dots

This is the proto-code.

It is not designed.
It emerges from substrate physics.

⸻

7. Transition Graph — The Proto Grammar

Transitions between attractors:

A_i \rightarrow A_j

form a directed graph.

Repeated transitions form:
	•	syntax
	•	rules
	•	operators
	•	compositional functions
	•	memory cycles
	•	branching structures

The transition graph is the early form of:
	•	grammar
	•	program
	•	computation

⸻

8. Emergent Hardware

The substrate gradually organizes into:
	•	hubs
	•	oscillators
	•	gating motifs
	•	long-range pathways
	•	slow-drift memory pockets
	•	feedback loops

This evolving topology is the hardware.

There is no separate “chip.”
Hardware is whatever physical structure repeatedly stabilizes under environmental pressure.

⸻

9. Phenotype: Acting Back on the Environment

The ultimate milestone is when the substrate:
	1.	performs computation
	2.	creates stable behavior
	3.	modifies its environment
	4.	which then affects its own future states

This forms a closed evolutionary cycle:

\text{substrate} \;\leftrightarrow\; \text{code} \;\leftrightarrow\; \text{environment}

This is the minimal definition of a phenotype in this framework.

⸻

10. Toward Universal Computation

The long-term objective is to show that Fuka-6.0 naturally evolves:
	1.	finite attractor alphabet
	2.	stable attractor sequences
	3.	compositional transition grammar
	4.	persistent multi-slot memory
	5.	gated read/write structures
	6.	branching transitions
	7.	feedback loops that represent functions

This combination yields the primitive conditions of a Turing-complete system emerging from physics alone.

⸻

11. Roadmap

📘 How Capacitors Work — and How Fuka Capacitors Compute

This section explains, in simple language, how real capacitors behave and how the Fuka-6.0 substrate uses a generalized capacitor model to create emergent symbols, code, and hardware.

### How a Real Capacitor Works
![Capacitor Physics Explained](images/3D3711EE-FB1C-4AEE-9352-DF266EB53D5C.png)


⸻

🧩 1. What is a capacitor?

A capacitor is the simplest device that can store and change electrical state.


It holds energy by separating charge.
Three important facts:

✔ It has a voltage

✔ It changes that voltage over time

✔ It stores energy in the electric field

The equations are:

Q = C V                  (charge = capacitance × voltage)
I = C dV/dt              (current changes voltage)
E = ½ C V²               (energy stored)


⸻

🧩 2. Why capacitors matter for computation

Capacitors naturally create:
	•	memory (stored voltage)
	•	dynamics (voltages evolve in time)
	•	attractors (stable voltage patterns)
	•	pattern separation (different states converge to different minima)

These are the same ingredients used by:
	•	neural networks
	•	analog computers
	•	Hopfield networks
	•	early biological systems

Capacitor networks naturally form state machines.

### How Fuka-6.0 Capacitors Work
![Fuka Capacitor Network](images/96FB08D3-A8E0-4225-9267-3B54A23906A5.png)

⸻

🧩 3. The Fuka-6.0 idea: A universe of capacitors

In Fuka-6.0, we generalize this idea.

We simulate a network of n abstract capacitors:

x = [x₁, x₂, x₃, ..., xₙ]

Each value xᵢ is the voltage of that capacitor at time t.

These capacitors interact through a conductance matrix:

g[i,j] = strength of coupling from capacitor j → i

This determines how charge “flows” between units.

### Transition Graph (Attractor Finite-State Machine)
![Transition Graph](images/ACEA2CE6-E90C-4611-8EC3-1918D595E02F.png)


⸻

🧩 4. What drives the capacitors?

There are three forces that change capacitor voltages.


⸻

(1) Internal dynamics (like charge flow)

Capacitors equalize through conductances:

Δxᵢ ∝ Σ gᵢⱼ ( xⱼ − xᵢ )

This creates:
	•	attractors
	•	stable patterns
	•	state convergence

These attractors eventually become symbols.

⸻

(2) External environment forcing

The environment (A, B, C or analog wave) pushes the system:

Δxᵢ ∝ α · E(t)

This is like an electrode injecting charge.

Environment → shapes the attractor basins → creates a consistent alphabet.

⸻

(3) Plasticity (rewiring the hardware)

Conductances change over time:
	•	connections strengthen
	•	unused paths decay
	•	modules form

This is how the substrate self-builds its own hardware.

In code, g is updated by local rules:

gₜ₊₁ = gₜ + f(local_state)

This is the heart of Fuka’s emergent hardware.

⸻

🧩 5. What encodes a symbol?

A symbol is not stored explicitly.

Instead:

✔ Symbols = attractor basins in state space

Example:

A = cluster of states near pattern pA
B = cluster near pB
C = cluster near pC

The substrate repeatedly falls into these patterns whenever the environment returns to the same regime.

This is how the alphabet emerges.

⸻

🧩 6. What encodes code?

Code emerges as the sequence of transitions between attractors.

Example:

A → B → B → A → ...

Each arrow is a directed transition in the attractor graph.

This graph is physically created by:
	•	the capacitor dynamics
	•	the conductance layout
	•	the influence of environment

This is equivalent to a proto grammar or a finite state machine.

⸻

🧩 7. What encodes hardware?

Hardware = the conductance matrix g.

This is the “wiring” of the substrate:

g =
[ g11 g12 g13 ... ]
[ g21 g22 g23 ... ]
[ ...            ]

Over time:
	•	g acquires structure
	•	modules appear
	•	repeated motifs emerge
	•	certain pathways become specialized

The substrate is literally building its own circuitry.

This is the link between:

physics → hardware → symbols → code → adaptation

⸻

🧩 8. Full mapping between physics and simulation

Real World	Fuka Capacitor Model	Meaning
Voltage	xᵢ	State/memory
Charge flow	Σ gᵢⱼ(xⱼ − xᵢ)	Interaction dynamics
External field	E(t)	Environment force
Cap geometry	plasticity	Hardware evolution
Energy minima	attractors	Symbols
State transitions	attractor shifts	Code
Circuit topology	conductance g	Hardware

This is the clean unification:

Capacitors → attractors → symbols → code → hardware → adaptation


⸻

🧩 9. Why this is important

This framework explains how:
	•	computation can emerge from physics
	•	symbols can emerge from pure dynamics
	•	hardware and code co-evolve
	•	adaptation becomes possible without pre-built structures
	•	biological systems may have originated

This is the conceptual foundation of Fuka-6.0.

⸻

Phase-6 Phenotype Results

Fuka-6.0 – Emergent alphabet, grammar, and substrate fingerprints

This section summarizes the Phase-6.1 phenotype run:

runs/exp_phenotype_fixed_20251129_230800.npz

From a single long run, the system produced a finite attractor alphabet, a probabilistic grammar over that alphabet, and stable physical “fingerprints” for each attractor. All of this emerges from local capacitor dynamics and plasticity, without any external supervision.

⸻

1. Core attractor alphabet

From 680 attractor samples, cosine clustering produced 293 distinct clusters.
If we keep only clusters that appear at least 10 times, we get a core alphabet of 9 attractors:

T1, T2, T3, T5, T8, T12, T14, T17, T24

Key statistics:
	•	Core alphabet coverage (clusters with count ≥ 10): 57.9% of all samples.
	•	The largest clusters have sizes: 95, 69, 61, 41, 33, 32, 32, 16, 15.
	•	The top 4 attractors alone cover roughly 40–45% of all samples.

This means the substrate settles repeatedly into a small, re-used set of attractor codes, rather than visiting all states uniformly. In the Fuka view, this set is the emergent symbolic alphabet of the phenotype.

⸻

2. Emergent phenotype grammar

We can build a Markov grammar over the core alphabet by looking at transitions from one core attractor to the next. The strongest core-to-core transitions are:
	•	T12 → T24 with probability ~0.73 (30 transitions)
	•	T1 → T17 with probability ~0.52 (32 transitions)
	•	T2 → T3 with probability ~0.45 (31 transitions)
	•	Self-loops such as T2 → T2 and T1 → T1
	•	Connector edges like T2 → T8, T3 → T5, T24 → T5

This defines a directed grammar where some attractors tend to follow others in a strongly biased way. The result is not a random walk. It is closer to a simple symbolic language with:
	•	Recurrent motifs (T12↔T24, T1→T17→T5, T2→T3→T5)
	•	Stable loops and hubs
	•	A small set of high-probability “grammar rules”

![Phenotype grammar](images/phenotype_grammar_latest.png)

Figure 1 – Core phenotype grammar graph

Figure 1: Graph of the core attractor alphabet. Nodes are the 9 most frequent attractors (T1, T2, T3, T5, T8, T12, T14, T17, T24). Directed edges show transitions between them. Edge thickness and labels encode transition probability and counts. The strongest motifs are T12 → T24, T1 → T17, and T2 → T3, indicating that the phenotype dynamics concentrate onto a small number of repeated symbolic patterns.

⸻

3. Environment-dependent syntax

The environment has a scalar state E(t) which is updated by the substrate and in turn modulates how strongly energy is injected. If we sample the environment value at the same times we take attractor snapshots and split by the median of E:
	•	Low-E band: more exploratory, almost no stable core-to-core transitions.
	•	High-E band: grammar becomes much more deterministic:
	•	T1 → T17 reaches probability 1.0 in high-E samples.
	•	T12 → T24 reaches probability ~0.94.
	•	T2 → T3 reaches probability ~0.63.

So when the environment is “energized”, the system collapses into a more rigid grammar. When energy is lower, it wanders and explores more codes.

In other words:

The scalar environment state controls how grammatical the behaviour is.
High energy = more strongly structured syntax.
Low energy = more diffuse, exploratory syntax.

⸻

4. Attractor fingerprints in substrate space

Each attractor can also be represented by its physical fingerprint: the mean voltage vector of the capacitor network whenever the system visits that attractor.

By projecting these fingerprints to 2D using PCA, we see that:
	•	Attractors that are grammatically connected (for example T12 and T24, or T1 and T17) tend to lie close to each other in fingerprint space.
	•	Attractors that rarely or never follow each other are more separated.

This means the symbolic grammar is not arbitrary. It is grounded in the actual physical geometry of the capacitor voltages.

![Attractor fingerprints](images/phenotype_fingerprints_latest.png)

Figure 2 – PCA map of core attractor fingerprints

Figure 2: PCA projection of the mean voltage fingerprints for the 9 core attractors. Each point is one attractor (T1, T2, T3, T5, T8, T12, T14, T17, T24). Attractors that often follow each other in the grammar tend to be neighbours in this space (for example T12 and T24). This links symbolic transitions directly to physical similarity in the capacitor substrate.

⸻

5. Interpretation

Putting everything together:
	•	A finite attractor alphabet emerges spontaneously from local physics.
	•	This alphabet carries a probabilistic grammar with strong repeated motifs.
	•	The grammar sharpens at high environment energy and weakens at low energy.
	•	Attractor identities are not abstract variables; they are bound to physical fingerprints in the capacitor network.

From the Fuka-6.0 point of view, this run is a first concrete example of:

A physical substrate that invents its own symbols, grammar, and phenotype-level behaviour from nothing more than energy flow, local plasticity, and a closed loop with its environment.


