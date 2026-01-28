const FIRESTORE_GET_URL = import.meta.env.VITE_FIRESTORE_GET_URL;

export default class FirestoreService {
  constructor() {}

  public async getFromFirestore() {
    const response = await fetch(FIRESTORE_GET_URL, {
      method: "GET",
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Firestore retrieval failed: ${text}`);
    }

    return await response.json();
  }
}
