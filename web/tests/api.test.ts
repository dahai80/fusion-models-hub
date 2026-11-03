import { describe, expect, test } from "bun:test";

import { apiBase } from "../src/api";

describe("apiBase (#88)", () => {
    test("embedded shell iframe returns the module proxy prefix", () => {
        expect(apiBase("/shell/fusion-models-hub/")).toBe("/shell/fusion-models-hub");
        expect(apiBase("/shell/fusion-models-hub")).toBe("/shell/fusion-models-hub");
    });

    test("nested module route keeps the prefix", () => {
        expect(apiBase("/shell/fusion-models-hub/index.html")).toBe("/shell/fusion-models-hub");
    });

    test("standalone hub origin returns empty base", () => {
        expect(apiBase("/")).toBe("");
        expect(apiBase("/index.html")).toBe("");
    });

    test("bare shell prefix without module id returns empty base", () => {
        expect(apiBase("/shell/")).toBe("");
        expect(apiBase("/shell")).toBe("");
    });
});
