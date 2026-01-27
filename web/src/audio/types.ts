export default interface firestoreRecord {
  created_at: string;
  transcript: string;
  moods: { label: string; score: number }[];
  mood_confidence: number;
  mood_evidence?: string[];
  uid: string;
}

export type TranscriptionMode = "stream" | "batch";
