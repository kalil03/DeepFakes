import { Shield } from "lucide-react";

const Header = () => {
  return (
    <header className="w-full py-4 px-6 flex items-center justify-between border-b border-[#1f1f1f] bg-[#0b0b0b]">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-md border border-[#1f1f1f] bg-[#121212] flex items-center justify-center">
          <Shield className="w-4 h-4 text-primary" />
        </div>
        <div className="leading-tight">
          <h1 className="text-sm font-semibold text-foreground">DeepTrace</h1>
          <p className="text-[11px] text-muted-foreground">
            Forensic deepfake detection for human faces
          </p>
        </div>
      </div>
      <span className="text-[11px] font-mono text-muted-foreground">
        v1.0 • Forensic mode
      </span>
    </header>
  );
};

export default Header;
