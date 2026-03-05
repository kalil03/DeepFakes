import { ShieldCheck, ShieldAlert, RefreshCcw } from "lucide-react";

export interface SightengineResult {
  is_deepfake: boolean;
  deepfake_score: number;
  is_ai_generated: boolean;
  ai_score: number;
  generator_type: "diffusion" | "gan" | "other" | "none";
  agreement?: "agree" | "disagree" | "unknown";
}

export interface ClassProbability {
  name: string;
  emoji: string;
  probability: number; // 0–1
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

  const se = result.sightengine;
  const hasSightengine = !!se;

  const seBadge =
    se && se.agreement === "agree"
      ? {
          label: "External verification: consistent",
          className:
            "text-xs font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
        }
      : se && se.agreement === "disagree"
        ? {
            label: "External verification: mismatch",
            className:
              "text-xs font-mono px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-200 border border-amber-500/40",
          }
        : null;

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
                  Predicted Source
                </p>
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <span className="text-lg">
                    {sortedProbs[0]?.emoji ?? "🧠"}
                  </span>
                  <span>{result.predictedClass}</span>
                </h3>
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
          Confidence by generator
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

      {/* Sightengine section */}
      {hasSightengine && se && (
        <div className="px-6 py-4 border-t border-[#1f1f1f] bg-[#050505]">
          <div className="flex items-center justify-between mb-2">
            <div>
              <p className="text-xs font-mono text-muted-foreground uppercase tracking-[0.25em]">
                External Verification
              </p>
              <p className="text-[11px] text-muted-foreground/80">
                Scores reported by Sightengine (independent model).
              </p>
            </div>
            {seBadge && (
              <span className={seBadge.className}>{seBadge.label}</span>
            )}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-2">
            <div className="rounded-lg bg-slate-900/70 border border-white/5 px-3 py-2">
              <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wide mb-0.5">
                Deepfake score
              </p>
              <p className="text-sm font-mono text-foreground">
                {(se.deepfake_score * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-lg bg-slate-900/70 border border-white/5 px-3 py-2">
              <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wide mb-0.5">
                AI score
              </p>
              <p className="text-sm font-mono text-foreground">
                {(se.ai_score * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-lg bg-slate-900/70 border border-white/5 px-3 py-2">
              <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wide mb-0.5">
                Generator type
              </p>
              <p className="text-sm font-mono text-foreground capitalize">
                {se.generator_type === "none"
                  ? "none"
                  : se.generator_type || "unknown"}
              </p>
            </div>
            <div className="rounded-lg bg-slate-900/70 border border-white/5 px-3 py-2">
              <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-wide mb-0.5">
                External verdict
              </p>
              <p className="text-sm font-mono text-foreground">
                {se.is_ai_generated || se.is_deepfake ? "AI / deepfake" : "Human"}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
