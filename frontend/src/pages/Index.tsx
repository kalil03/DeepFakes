import { useState, useCallback } from "react";
import Header from "@/components/Header";
import ImageUpload from "@/components/ImageUpload";
import AnalyzeButton from "@/components/AnalyzeButton";
import ResultCard, {
  type AnalysisResult,
  type ClassProbability,
  type SightengineResult,
} from "@/components/ResultCard";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const { toast } = useToast();

  const handleImageSelect = useCallback(
    (file: File) => {
      // Validate file size (10 MB max, matching backend limit)
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Maximum file size is 10 MB.",
          variant: "destructive",
        });
        return;
      }

      // Revoke previous URL to prevent memory leak
      if (imagePreview) URL.revokeObjectURL(imagePreview);

      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
    },
    [imagePreview, toast],
  );

  const handleImageRemove = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImage(null);
    setImagePreview(null);
    setResult(null);
  }, [imagePreview]);

  const handleReset = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImage(null);
    setImagePreview(null);
    setResult(null);
  }, [imagePreview]);

  const handleAnalyze = useCallback(async () => {
    if (!image) return;
    setIsAnalyzing(true);
    setResult(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json().catch(() => ({}));

      if (!response.ok || data.error) {
        throw new Error(
          data.error || data.detail || `HTTP error ${response.status}`,
        );
      }

      const {
        predicted_class,
        confidence,
        probabilities,
        uncertain,
        sightengine,
      } = data as {
        predicted_class: string;
        confidence: number;
        probabilities: Record<string, number>;
        uncertain: boolean;
        sightengine?: SightengineResult;
      };

      const emojiMap: Record<string, string> = {
        "Human (Real)": "👤",
        "Deepfake (GAN)": "🤖",
      };

      const classProbs: ClassProbability[] = Object.entries(probabilities).map(
        ([name, prob]) => ({
          name,
          emoji: emojiMap[name] ?? "🧠",
          probability: prob,
        }),
      );

      // Sightengine agreement heuristic
      let seWithAgreement: SightengineResult | undefined;
      if (sightengine) {
        const localIsAi = predicted_class !== "Human (Real)";
        const seIsAi = sightengine.is_ai_generated || sightengine.is_deepfake;
        const agreement =
          seIsAi === localIsAi ? ("agree" as const) : ("disagree" as const);
        seWithAgreement = { ...sightengine, agreement };
      }

      const finalResult: AnalysisResult = {
        predictedClass: predicted_class,
        confidence,
        uncertain,
        probabilities: classProbs,
        sightengine: seWithAgreement,
      };

      setResult(finalResult);
    } catch (error) {
      console.error("Inference Error:", error);
      toast({
        title: "Something went wrong",
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
      clearTimeout(timeoutId);
    }
  }, [image, toast]);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 flex flex-col items-center justify-center px-4 py-10 gap-6">
        <div className="w-full max-w-xl text-left mb-2">
          <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-foreground mb-1">
            Is this face real?
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-xl">
            Upload a portrait or face photo to detect if it was generated or
            manipulated by AI.
          </p>
        </div>

        <ImageUpload
          image={image}
          imagePreview={imagePreview}
          onImageSelect={handleImageSelect}
          onImageRemove={handleImageRemove}
          isAnalyzing={isAnalyzing}
        />

        <p className="text-[11px] text-muted-foreground max-w-xl text-left">
          ⚠️ Optimized for face/portrait images only.
        </p>

        <AnalyzeButton
          onClick={handleAnalyze}
          isAnalyzing={isAnalyzing}
          disabled={!image}
        />

        {result && (
          <ResultCard
            result={result}
            imagePreview={imagePreview}
            onReset={handleReset}
          />
        )}
      </main>

      <footer className="py-4 text-center">
        <p className="text-xs font-mono text-muted-foreground">
          DeepTrace — Forensic deepfake detection for human faces
        </p>
      </footer>
    </div>
  );
};

export default Index;
