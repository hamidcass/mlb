
// TODO: change to deployment URL when deploying
const BASE_URL = "http://127.0.0.1:8000";

export async function fetchPlayer(playerName: string) {
    const res = await fetch(`${BASE_URL}/player/${encodeURIComponent(playerName)}`);
    return res.json();

}

export async function fetchPredictions(stat: string, model: string, limit = 20) {
    const res = await fetch(`${BASE_URL}/predictions?stat=${stat}&model=${model}&limit=${limit}`);
    return res.json();
}

export async function fetchAllPredictions(stat: string, model: string) {
    const res = await fetch(`${BASE_URL}/predictions?stat=${stat}&model=${model}`);
    if (!res.ok) throw new Error("Failed to fetch predictions");
    return res.json();
}

export async function fetchMeta() {
    const res = await fetch(`${BASE_URL}/meta`);
    if (!res.ok) throw new Error("Failed to fetch meta");
    return res.json();
}

export async function fetchMetrics(stat: string, model: string) {
    const res = await fetch(`${BASE_URL}/metrics?stat=${stat}&model=${model}`);
    if (!res.ok) throw new Error("Failed to fetch metrics");
    return res.json();
}

export async function fetchImportance(stat: string, model: string) {
    const res = await fetch(`${BASE_URL}/importance?stat=${stat}&model=${model}`);
    if (!res.ok) throw new Error("Failed to fetch importance");
    return res.json();
}

export async function fetchStats() {
    const res = await fetch(`${BASE_URL}/stats`);
    if (!res.ok) throw new Error("Failed to fetch stats");
    return res.json();
}