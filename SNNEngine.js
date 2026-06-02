/**
 * SNN Simulation Engine - Coordinate-based ID Version
 */
export class SNNEngine {
  constructor(model) {
    this.model = model;
    this.states = new Map();
    this.reset();
  }

  reset() {
    this.model.nodes.forEach(node => {
      this.states.set(node.id, {
        v: 0.0,
        spike: 0.0,
        buffer: new Array(15).fill(0.0),
        refrac_abs: 0
      });
    });
  }

  step() {
    // Helper to update a single node's state
    const updateNode = (node) => {
      let state = this.states.get(node.id);
      let arrivingInput = state.buffer.shift();
      state.buffer.push(0.0);
      state.v += arrivingInput;

      if (state.v > 0) {
        state.v *= 0.9;
      } else {
        state.v = Math.min(0, state.v + 0.1);
      }

      if (state.refrac_abs > 0) {
        state.refrac_abs--;
        state.spike = 0.0;
      } else if (state.v >= node.threshold) {
        state.spike = 1.0;
        state.v = -0.5;
        state.refrac_abs = 2;
      } else {
        state.spike = 0.0;
      }
    };

    // Helper to deliver spikes from a set of source node IDs
    const deliverSpikes = (sourceNodeIds) => {
      this.model.edges.forEach(edge => {
        if (sourceNodeIds.has(edge.source)) {
          let srcState = this.states.get(edge.source);
          if (srcState.spike > 0) {
            let tgtState = this.states.get(edge.target);
            let delayIdx = Math.min(edge.delay - 1, tgtState.buffer.length - 1);
            if (delayIdx < 0) delayIdx = 0;
            tgtState.buffer[delayIdx] += (edge.sign * edge.weight);
          }
        }
      });
    };

    // Split nodes into Inhibitory and Excitatory
    const inhibNodes = this.model.nodes.filter(n => !n.excitatory);
    const excitNodes = this.model.nodes.filter(n => n.excitatory);

    const inhibNodeIds = new Set(inhibNodes.map(n => n.id));
    const excitNodeIds = new Set(excitNodes.map(n => n.id));

    // Phase A: Inhibitory updates + immediate spike delivery
    inhibNodes.forEach(updateNode);
    deliverSpikes(inhibNodeIds);

    // Phase B: Excitatory updates + spike delivery
    excitNodes.forEach(updateNode);
    deliverSpikes(excitNodeIds);

    const snapshot = {};
    for (const [id, s] of this.states) snapshot[id] = s.spike;
    return snapshot;
  }

  predict(inputValues, ticks = 10) {
    this.reset();
    const inputNodes = this.model.nodes.filter(n => n.layer === 0);
    inputValues.forEach((val, i) => {
      if (val > 0 && inputNodes[i]) {
        this.states.get(inputNodes[i].id).v = inputNodes[i].threshold * 2.0;
      }
    });
    let history = [];
    for (let i = 0; i < ticks; i++) {
      history.push(this.step());
    }
    return history;
  }
}
