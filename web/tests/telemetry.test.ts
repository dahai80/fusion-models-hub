import { describe, expect, test } from "bun:test";
import {
    type ByteSample,
    SpeedTracker,
    computeEtaMs,
    computeSpeedBps,
    downloadSource,
} from "../src/telemetry";

const T0 = 1700000000000;

describe("computeSpeedBps", () => {
    test("null with fewer than 2 samples", () => {
        expect(computeSpeedBps([{ at: T0, bytes: 100 }], T0)).toBeNull();
        expect(computeSpeedBps([], T0)).toBeNull();
    });

    test("null with zero elapsed time", () => {
        const samples: ByteSample[] = [
            { at: T0, bytes: 100 },
            { at: T0, bytes: 200 },
        ];
        expect(computeSpeedBps(samples, T0)).toBeNull();
    });

    test("null on byte counter reset (resume/rollback)", () => {
        const samples: ByteSample[] = [
            { at: T0, bytes: 500 },
            { at: T0 + 5000, bytes: 100 },
        ];
        expect(computeSpeedBps(samples, T0 + 5000)).toBeNull();
    });

    test("average over window", () => {
        const samples: ByteSample[] = [
            { at: T0, bytes: 0 },
            { at: T0 + 2000, bytes: 2000 },
            { at: T0 + 4000, bytes: 4000 },
        ];
        expect(computeSpeedBps(samples, T0 + 4000)).toBe(1000);
    });

    test("samples outside the 15s window are excluded", () => {
        const samples: ByteSample[] = [
            { at: T0, bytes: 0 },
            { at: T0 + 20000, bytes: 10000 },
            { at: T0 + 30000, bytes: 30000 },
        ];
        expect(computeSpeedBps(samples, T0 + 30000)).toBe(2000);
    });
});

describe("computeEtaMs", () => {
    test("null without speed", () => {
        expect(computeEtaMs(null, 1000)).toBeNull();
    });

    test("null with non-positive speed", () => {
        expect(computeEtaMs(0, 1000)).toBeNull();
    });

    test("null with nothing remaining", () => {
        expect(computeEtaMs(1000, 0)).toBeNull();
    });

    test("remaining bytes over speed", () => {
        expect(computeEtaMs(1000, 5000)).toBe(5000);
    });
});

describe("downloadSource", () => {
    test("hf-mirror.com detected case-insensitively", () => {
        expect(downloadSource("https://hf-mirror.com/Qwen/Qwen2.5-0.5B")).toBe(
            "HF_MIRROR",
        );
        expect(downloadSource("https://HF-MIRROR.com/org/repo")).toBe(
            "HF_MIRROR",
        );
    });

    test("huggingface.co", () => {
        expect(downloadSource("https://huggingface.co/org/repo")).toBe(
            "HUGGINGFACE",
        );
    });

    test("modelscope", () => {
        expect(downloadSource("https://modelscope.cn/models/qwen/Qwen2.5")).toBe(
            "MODELSCOPE",
        );
    });

    test("anything else", () => {
        expect(downloadSource("http://192.168.1.10:8080/model.bin")).toBe(
            "OTHER",
        );
    });
});

describe("SpeedTracker", () => {
    test("records and derives speed per task", () => {
        const tracker = new SpeedTracker();
        tracker.record("t1", T0, 0);
        tracker.record("t1", T0 + 1000, 1024);
        expect(tracker.speed("t1", T0 + 1000)).toBe(1024);
    });

    test("unknown task has no speed", () => {
        expect(new SpeedTracker().speed("nope")).toBeNull();
    });

    test("reset clears samples", () => {
        const tracker = new SpeedTracker();
        tracker.record("t1", T0, 0);
        tracker.record("t1", T0 + 1000, 1024);
        tracker.reset("t1");
        expect(tracker.speed("t1", T0 + 1000)).toBeNull();
    });

    test("sample buffer is bounded", () => {
        const tracker = new SpeedTracker();
        for (let i = 0; i < 50; i++) {
            tracker.record("t1", T0 + i * 100, i * 10);
        }
        // Bounded to MAX_SAMPLES (30): speed still computes from the window.
        expect(tracker.speed("t1", T0 + 4900)).toBe(100);
    });
});
