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
      className="relative group px-7 py-3 rounded-md font-medium text-xs sm:text-sm tracking-wide transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed bg-primary text-primary-foreground hover:bg-primary/90"
    >
      <span className="flex items-center gap-2.5">
        {isAnalyzing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing…
          </>
        ) : (
          <>
            <Scan className="w-4 h-4" />
            Analyze Image
          </>
        )}
      </span>
    </button>
  );
};

export default AnalyzeButton;
