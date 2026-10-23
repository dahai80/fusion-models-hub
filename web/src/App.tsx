import { useEffect, useState } from "react";
import { DownloadsPage } from "./pages/DownloadsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { QuantizePage } from "./pages/QuantizePage";
import { StoragePage } from "./pages/StoragePage";
import { embeddedParam, isEmbedded } from "./sdk";

// Hash router (no router dep). Standalone shows a bottom tab bar; embedded
// (?embedded=1 or iframe) the shell owns the chrome (spec S7).
type Route = "models" | "downloads" | "quantize" | "storage";

const ROUTES: ReadonlySet<string> = new Set<Route>([
    "models",
    "downloads",
    "quantize",
    "storage",
]);

function parseHash(): Route {
    const h = window.location.hash.replace(/^#\/?/, "");
    if (ROUTES.has(h)) return h as Route;
    return "models";
}

export function App() {
    const [route, setRoute] = useState<Route>(parseHash());

    useEffect(() => {
        const onHash = () => setRoute(parseHash());
        window.addEventListener("hashchange", onHash);
        return () => window.removeEventListener("hashchange", onHash);
    }, []);

    const embedded = embeddedParam() || isEmbedded();

    return (
        <div className={embedded ? "h-full" : "flex h-full flex-col"}>
            {embedded && <EmbeddedTabs route={route} onChange={setRoute} />}
            <main className={`min-h-0 flex-1 ${embedded ? "" : "pb-16 pt-4"}`}>
                {route === "models" && <ModelsPage />}
                {route === "downloads" && <DownloadsPage />}
                {route === "quantize" && <QuantizePage />}
                {route === "storage" && <StoragePage />}
            </main>
            {!embedded && <BottomNav route={route} onChange={setRoute} />}
        </div>
    );
}

const NAV_ITEMS: Array<{ id: Route; label: string }> = [
    { id: "models", label: "Models" },
    { id: "downloads", label: "Downloads" },
    { id: "quantize", label: "Quantize" },
    { id: "storage", label: "Storage" },
];

function EmbeddedTabs({
    route,
    onChange,
}: { route: Route; onChange: (r: Route) => void }) {
    return (
        <nav className="flex shrink-0 border-b border-border bg-bg-elevated">
            {NAV_ITEMS.map((it) => (
                <button
                    key={it.id}
                    type="button"
                    onClick={() => onChange(it.id)}
                    className={`flex-1 py-2 text-center text-xs uppercase tracking-wide ${
                        route === it.id
                            ? "border-b-2 border-brand text-brand"
                            : "text-text-secondary"
                    }`}
                >
                    {it.label}
                </button>
            ))}
        </nav>
    );
}

function BottomNav({
    route,
    onChange,
}: { route: Route; onChange: (r: Route) => void }) {
    return (
        <nav className="fixed inset-x-0 bottom-0 z-10 flex border-t border-border bg-bg-elevated">
            {NAV_ITEMS.map((it) => (
                <button
                    key={it.id}
                    type="button"
                    onClick={() => onChange(it.id)}
                    className={`flex-1 py-3 text-center text-sm ${
                        route === it.id ? "text-brand" : "text-text-secondary"
                    }`}
                >
                    {it.label}
                </button>
            ))}
        </nav>
    );
}
