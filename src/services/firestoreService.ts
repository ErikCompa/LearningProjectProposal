export default class FirestoreService {
  constructor() {}

  public async getFromFirestore() {
    const response = await fetch("http://localhost:8000/v1/firestore_get/", {
      method: "GET",
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Firestore retrieval failed: ${text}`);
    }

    return await response.json();
  }
}
