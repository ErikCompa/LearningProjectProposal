// https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor
class RecorderNode extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];

    if (inputs.length > 0) {
      this.port.postMessage(input);
    }
    return true;
  }
}

registerProcessor("recorder-node", RecorderNode);
