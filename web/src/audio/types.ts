export interface Mood {
  label: string;
  score: number;
}

export interface Top50Mood {
  label: string;
  score: number;
}

export interface Top50Evidence {
  label: string;
  explanation: string;
}

export default interface firestoreRecord {
  created_at: string;
  transcript: string;
  moods: Mood[];
  mood_confidence: number;
  mood_evidence?: string[];
  uid: string;
  top_50_moods?: Top50Mood[];
  top_50_evidences?: Top50Evidence[];
}

export type TranscriptionMode = "stream" | "batch";
