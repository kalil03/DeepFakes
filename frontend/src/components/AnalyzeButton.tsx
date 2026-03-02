import { Loader2, Scan } from "lucide-react";

interface AnalyzeButtonProps {
  onClick: () => void;
  isAnalyzing: boolean;
  disabled: boolean;
}

const AnalyzeButton = ({ onClick, isAnalyzing, disabled }: AnalyzeButtonProps) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isAnalyzing}
      className="relative group px-8 py-3.5 rounded-xl font-semibold text-sm tracking-wide transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed gradient-primary text-primary-foreground hover:shadow-[0_0_30px_-5px_hsl(var(--glow-primary)/0.5)] active:scale-[0.98]"
    >
      <span className="flex items-center gap-2.5">
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analisando...
          </>
        ) : (
          <>
            <Scan className="w-4 h-4" />
            Analisar Imagem
          </>
        )}
      </span>
    </button>
  );
};

export default AnalyzeButton;
