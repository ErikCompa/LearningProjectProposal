const BATCH_AUDIO_URL = import.meta.env.VITE_BATCH_AUDIO_URL;

export default class BatchService {
  public async processBatchAudio(
    linear16Buffer: Int16Array,
    provider: string = "google",
  ): Promise<void> {
    try {
      // convert into audio blob
      const arrayBuffer = new ArrayBuffer(linear16Buffer.byteLength);
      new Uint8Array(arrayBuffer).set(new Uint8Array(linear16Buffer.buffer));
      var blob = new Blob([arrayBuffer], { type: "application/octet-stream" });

      // append blob to form as wav file
      const form = new FormData();
      form.append("file", blob, "speech.wav");

      const url = `${BATCH_AUDIO_URL}?provider=${provider}`;

      const res = await fetch(url, {
        method: "POST",
        body: form,
      });

      if (!res.status || res.status !== 200) {
        const text = await res.text();
        throw new Error(`Processing audio failed: ${text}`);
      }

      await res.json();
    } catch (err) {
      console.error("Error processing audio:", err);
      return;
    }
  }
}
