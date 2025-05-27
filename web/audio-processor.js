class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.buffer = new Float32Array(0);
      this.port.onmessage = (e) => {
        const newData = e.data;
        this.buffer = this.mergeBuffers(this.buffer, newData);
      };
    }
  
    process(inputs, outputs) {
      const output = outputs[0][0];
      if (this.buffer.length < output.length) {
        output.fill(0);
        return true;
      }
      
      output.set(this.buffer.subarray(0, output.length));
      this.buffer = this.buffer.subarray(output.length);
      return true;
    }
  
    mergeBuffers(oldBuffer, newBuffer) {
      const merged = new Float32Array(oldBuffer.length + newBuffer.length);
      merged.set(oldBuffer);
      merged.set(newBuffer, oldBuffer.length);
      return merged;
    }
  }
  
  registerProcessor('pcm-processor', PCMProcessor);