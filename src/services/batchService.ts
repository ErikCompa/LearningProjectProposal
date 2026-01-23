export default class BatchService {
  public async processBatchAudio(linear16Buffer: Int16Array): Promise<void> {
    try {
      // convert into audio blob
      const arrayBuffer = new ArrayBuffer(linear16Buffer.byteLength);
      new Uint8Array(arrayBuffer).set(new Uint8Array(linear16Buffer.buffer));
      var blob = new Blob([arrayBuffer], { type: "application/octet-stream" });

      // append blob to form as wav file
      const form = new FormData();
      form.append("file", blob, "speech.wav");

      const res = await fetch("http://localhost:8000/v1/batch_process_audio/", {
        method: "POST",
        body: form,
      });

      if (!res.status || res.status !== 200) {
        const text = await res.text();
        throw new Error(`Processing audio failed: ${text}`);
      }
    } catch (err) {
      console.error("Error processing audio:", err);
      return;
    }
  }
}
