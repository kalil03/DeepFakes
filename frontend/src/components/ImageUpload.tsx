import { useCallback } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";

interface ImageUploadProps {
  image: File | null;
  imagePreview: string | null;
  onImageSelect: (file: File) => void;
  onImageRemove: () => void;
  isAnalyzing: boolean;
}

const ImageUpload = ({ image, imagePreview, onImageSelect, onImageRemove, isAnalyzing }: ImageUploadProps) => {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onImageSelect(file);
  };

  if (imagePreview) {
    return (
      <div className="relative w-full max-w-lg mx-auto rounded-2xl overflow-hidden glass glow-border group">
        <img
          src={imagePreview}
          alt="Uploaded"
          className="w-full h-72 object-contain"
        />
        {isAnalyzing && (
          <div className="absolute inset-0 bg-background/60 flex items-center justify-center">
            <div className="absolute inset-0 overflow-hidden">
              <div className="w-full h-1 gradient-primary scan-line" />
            </div>
          </div>
        )}
        {!isAnalyzing && (
          <button
            onClick={onImageRemove}
            className="absolute top-3 right-3 w-8 h-8 rounded-full bg-background/80 backdrop-blur flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
        <div className="absolute bottom-0 inset-x-0 p-3 bg-gradient-to-t from-background/90 to-transparent">
          <p className="text-xs font-mono text-muted-foreground truncate">{image?.name}</p>
        </div>
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className="relative w-full max-w-lg mx-auto"
    >
      <label
        htmlFor="image-upload"
        className="flex flex-col items-center justify-center w-full h-60 rounded-lg border border-dashed border-[#1f1f1f] bg-[#101010] cursor-pointer transition-colors duration-150 group"
      >
        <div className="w-12 h-12 rounded-md border border-[#1f1f1f] bg-[#141414] flex items-center justify-center mb-3">
          <Upload className="w-7 h-7 text-primary" />
        </div>
        <p className="text-xs sm:text-sm font-medium text-foreground mb-1">
          Drag an image here or{" "}
          <span className="text-primary font-semibold">click to select</span>
        </p>
        <p className="text-[11px] text-muted-foreground font-mono">
          PNG, JPG, WEBP • up to 10MB
        </p>
        <div className="flex items-center gap-1.5 mt-3 text-muted-foreground">
          <ImageIcon className="w-3.5 h-3.5" />
          <span className="text-[11px] font-mono">
            Supported formats: jpg, png, webp
          </span>
        </div>
      </label>
      <input
        id="image-upload"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  );
};

export default ImageUpload;
