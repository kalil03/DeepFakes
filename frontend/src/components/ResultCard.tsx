import { ShieldCheck, ShieldAlert } from "lucide-react";

export interface AnalysisResult {
  isReal: boolean;
  confidence: number;
  categories: {
    human: number;
    deepfakeGan: number;
    dalle3: number;
    midjourneyV6: number;
    stableDiffusion: number;
    googleGemini: number;
  };
}

interface ResultCardProps {
  result: AnalysisResult;
}

const categoryLabels: { key: keyof AnalysisResult["categories"]; label: string; tag: string }[] = [
  { key: "human", label: "Humano (Real)", tag: "REAL" },
  { key: "deepfakeGan", label: "Deepfake GAN", tag: "GAN" },
  { key: "dalle3", label: "DALL·E 3", tag: "DALL-E" },
  { key: "midjourneyV6", label: "Midjourney v6", tag: "MJ" },
  { key: "stableDiffusion", label: "Stable Diffusion", tag: "SD" },
  { key: "googleGemini", label: "Google Gemini", tag: "GEMINI" },
];

const ResultCard = ({ result }: ResultCardProps) => {
  const isReal = result.isReal;

  return (
    <div className="w-full max-w-lg mx-auto rounded-2xl glass-strong glow-border overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Verdict */}
      <div className={`px-6 py-5 flex items-center gap-4 border-b border-white/[0.06] ${isReal ? "bg-[hsl(var(--safe)/0.08)]" : "bg-[hsl(var(--danger)/0.08)]"}`}>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${isReal ? "bg-[hsl(var(--safe)/0.15)]" : "bg-[hsl(var(--danger)/0.15)]"}`}>
          {isReal ? (
            <ShieldCheck className="w-6 h-6 text-safe" />
          ) : (
            <ShieldAlert className="w-6 h-6 text-danger" />
          )}
        </div>
        <div>
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-0.5">
            Veredito
          </p>
          <h3 className={`text-xl font-bold ${isReal ? "text-safe" : "text-danger"}`}>
            {isReal ? "Humano (Real)" : "Gerado por IA"}
          </h3>
        </div>
      </div>

      {/* Overall Confidence */}
      <div className="px-6 py-4 border-b border-white/[0.06]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
            Confiança Geral
          </span>
          <span className="text-sm font-bold font-mono text-foreground">
            {result.confidence.toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-2.5 rounded-full bg-secondary overflow-hidden">
          <div
            className={isReal ? "progress-bar-safe" : "progress-bar-danger"}
            style={{ width: `${result.confidence}%` }}
          />
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="px-6 py-5 space-y-4">
        <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          Atribuição por Modelo
        </p>
        {categoryLabels.map(({ key, label, tag }) => {
          const value = result.categories[key];
          const isTop = value === Math.max(...Object.values(result.categories));
          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${isTop ? "gradient-primary text-primary-foreground" : "bg-secondary text-muted-foreground"}`}>
                    {tag}
                  </span>
                  <span className={`text-sm ${isTop ? "text-foreground font-semibold" : "text-muted-foreground"}`}>
                    {label}
                  </span>
                </div>
                <span className={`text-sm font-mono font-bold ${isTop ? "text-foreground" : "text-muted-foreground"}`}>
                  {value.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className={key === "human" ? "progress-bar-safe" : "progress-bar-fill"}
                  style={{ width: `${value}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ResultCard;
