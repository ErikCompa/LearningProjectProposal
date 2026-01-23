export default interface FirestoreRecord {
  uid: string;
  transcript: string;
  mood: {
    mood: string;
    confidence: number;
    evidence?: string[] | null;
  };
  created_at: string;
}

export type TranscriptionMode = "stream" | "batch";
