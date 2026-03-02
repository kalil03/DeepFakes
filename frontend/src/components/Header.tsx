import { Shield, Scan } from "lucide-react";

const Header = () => {
  return (
    <header className="w-full py-5 px-6 flex items-center justify-between glass-strong border-b border-white/[0.06]">
      <div className="flex items-center gap-3">
        <div className="relative w-10 h-10 rounded-lg gradient-primary flex items-center justify-center">
          <Shield className="w-5 h-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-lg font-bold tracking-tight text-foreground">
            Veritas <span className="gradient-text">AI</span>
          </h1>
          <p className="text-[11px] font-mono text-muted-foreground tracking-widest uppercase">
            Deepfake Detection
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2 text-muted-foreground">
        <Scan className="w-4 h-4 text-primary" />
        <span className="text-xs font-mono">v1.0</span>
      </div>
    </header>
  );
};

export default Header;
