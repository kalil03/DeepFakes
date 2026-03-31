import { ShieldCheck, ShieldAlert, RefreshCcw } from "lucide-react";

export interface ClassProbability {
  name: string;
  emoji: string;
  probability: number; // 0–1
}

export interface SightengineResult {
  is_deepfake: boolean;
  deepfake_score: number;
  is_ai_generated: boolean;
  ai_score: number;
  generator_type: string;
  agreement?: "agree" | "disagree";
}

export interface AnalysisResult {
  predictedClass: string;
  confidence: number; // 0–1
  uncertain: boolean;
  probabilities: ClassProbability[];
  sightengine?: SightengineResult;
}

interface ResultCardProps {
  result: AnalysisResult;
  imagePreview: string | null;
  onReset: () => void;
}

const ResultCard = ({ result, imagePreview, onReset }: ResultCardProps) => {
  const isReal = result.predictedClass === "Human (Real)";

  const sortedProbs = [...result.probabilities].sort(
    (a, b) => b.probability - a.probability,
  );

  return (
    <div className="w-full max-w-3xl mx-auto rounded-xl border border-[#1f1f1f] bg-[#050505] overflow-hidden animate-in fade-in duration-200">
      {/* Top: image preview + main verdict */}
      <div className="grid md:grid-cols-[1.2fr,1.5fr] gap-px border-b border-white/10 bg-border/40">
        {/* Image preview */}
        <div className="bg-background p-4 flex flex-col gap-3">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-[0.25em]">
            Analyzed Image
          </p>
          <div className="relative rounded-md overflow-hidden border border-[#1f1f1f] bg-black/40 min-h-[180px] flex items-center justify-center">
            {imagePreview ? (
              <img
                src={imagePreview}
                alt="Analyzed"
                className="w-full h-full object-contain"
              />
            ) : (
              <span className="text-xs text-muted-foreground">
                No image preview available
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center justify-center gap-2 text-xs font-medium px-3 py-2 rounded-lg bg-secondary/80 hover:bg-secondary text-foreground transition-colors mt-1 self-start"
          >
            <RefreshCcw className="w-3 h-3" />
            Analyze another
          </button>
        </div>

        {/* Verdict + overall confidence */}
        <div
          className={`px-6 py-5 flex flex-col gap-4 ${
            isReal
              ? "bg-[hsl(var(--safe)/0.08)]"
              : "bg-[hsl(var(--danger)/0.08)]"
          }`}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              <div
                className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  isReal
                    ? "bg-[hsl(var(--safe)/0.15)]"
                    : "bg-[hsl(var(--danger)/0.15)]"
                }`}
              >
                {isReal ? (
                  <ShieldCheck className="w-6 h-6 text-safe" />
                ) : (
                  <ShieldAlert className="w-6 h-6 text-danger" />
                )}
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-0.5">
                  Predicted verdict
                </p>
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <span className="text-lg">
                    {sortedProbs[0]?.emoji ?? "🧠"}
                  </span>
                  <span>{result.predictedClass}</span>
                </h3>
                <p className="text-xs text-muted-foreground mt-1">
                  {isReal
                    ? "No manipulation detected."
                    : "AI generation or manipulation detected."}
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                Overall confidence
              </span>
              <span className="text-sm font-bold font-mono text-foreground">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2.5 rounded-full bg-secondary overflow-hidden">
              <div
                className={
                  isReal ? "progress-bar-safe" : "progress-bar-danger"
                }
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>

          {result.uncertain && (
            <div className="mt-1 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 flex items-start gap-2">
              <span className="text-lg leading-none">⚠️</span>
              <div className="text-xs text-amber-100 text-left">
                <p className="font-semibold mb-0.5">
                  Low confidence — result may be inaccurate
                </p>
                <p className="text-[11px] text-amber-100/80">
                  Try a higher resolution image or a clearer crop of the main
                  subject.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Class probability breakdown */}
      <div className="px-6 py-5 space-y-4 bg-background">
        <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          Confidence by class
        </p>
        <div className="space-y-3">
          {sortedProbs.map((item, idx) => {
            const pct = item.probability * 100;
            const isTop = idx === 0;
            return (
              <div key={item.name}>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{item.emoji}</span>
                    <span
                      className={`text-sm ${
                        isTop
                          ? "text-foreground font-semibold"
                          : "text-muted-foreground"
                      }`}
                    >
                      {item.name}
                    </span>
                  </div>
                  <span
                    className={`text-sm font-mono font-bold ${
                      isTop ? "text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {pct.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className={
                      item.name === "Human (Real)"
                        ? "progress-bar-safe"
                        : "progress-bar-fill"
                    }
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Sightengine cross-validation (if available) */}
      {result.sightengine && (
        <div className="px-6 py-4 border-t border-[#1f1f1f] bg-background">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-3">
            Sightengine Cross-Check
          </p>
          <div className="flex items-center gap-3 text-xs">
            <span
              className={`px-2 py-1 rounded font-medium ${
                result.sightengine.agreement === "agree"
                  ? "bg-emerald-500/15 text-emerald-400"
                  : "bg-amber-500/15 text-amber-400"
              }`}
            >
              {result.sightengine.agreement === "agree"
                ? "✓ Models agree"
                : "⚠ Models disagree"}
            </span>
            <span className="text-muted-foreground">
              AI score: {(result.sightengine.ai_score * 100).toFixed(0)}% •
              Deepfake score:{" "}
              {(result.sightengine.deepfake_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
