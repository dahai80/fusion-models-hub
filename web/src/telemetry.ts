// Download telemetry helpers (spec §7.3): the hub API reports cumulative
// bytes only — speed and ETA are derived client-side from successive poll
// samples. Pure functions so they are unit-testable without a server.

export interface ByteSample {
    at: number;
    bytes: number;
}

export const SPEED_WINDOW_MS = 15000;

// Average bytes/second over the sample window (oldest vs newest sample in
// the window). Returns null when there are fewer than 2 samples or the
// window has no elapsed time (avoids Infinity on a fast poll loop).
export function computeSpeedBps(
    samples: ByteSample[],
    now: number = Date.now(),
): number | null {
    const windowed = samples.filter((s) => now - s.at <= SPEED_WINDOW_MS);
    if (windowed.length < 2) return null;
    const first = windowed[0];
    const last = windowed[windowed.length - 1];
    const dtMs = last.at - first.at;
    if (dtMs <= 0) return null;
    const db = last.bytes - first.bytes;
    if (db < 0) return null;
    return (db / dtMs) * 1000;
}

// Remaining-time estimate in ms from the current speed. Null when speed is
// missing, non-positive, or there is nothing left to download.
export function computeEtaMs(
    speedBps: number | null,
    remainingBytes: number,
): number | null {
    if (speedBps === null || speedBps <= 0) return null;
    if (remainingBytes <= 0) return null;
    return (remainingBytes / speedBps) * 1000;
}

export type DownloadSource =
    | "HF_MIRROR"
    | "HUGGINGFACE"
    | "MODELSCOPE"
    | "OTHER";

// Source badge (spec §7.3): HF_MIRROR downloads are the ecosystem default —
// make the origin visible so users know which mirror served the bytes.
export function downloadSource(sourceUrl: string): DownloadSource {
    const url = sourceUrl.toLowerCase();
    if (url.includes("hf-mirror.com")) return "HF_MIRROR";
    if (url.includes("huggingface.co")) return "HUGGINGFACE";
    if (url.includes("modelscope.cn") || url.includes("modelscope.com")) {
        return "MODELSCOPE";
    }
    return "OTHER";
}

// Per-task sample tracker: keeps a bounded sliding window per download task
// across polls. Cleared for terminal tasks so a retried id starts fresh.
const MAX_SAMPLES = 30;

export class SpeedTracker {
    private samples = new Map<string, ByteSample[]>();

    record(taskId: string, at: number, bytes: number): void {
        const list = this.samples.get(taskId) ?? [];
        list.push({ at, bytes });
        while (list.length > MAX_SAMPLES) list.shift();
        this.samples.set(taskId, list);
    }

    speed(taskId: string, now: number = Date.now()): number | null {
        const list = this.samples.get(taskId);
        if (!list) return null;
        return computeSpeedBps(list, now);
    }

    reset(taskId: string): void {
        this.samples.delete(taskId);
    }
}
