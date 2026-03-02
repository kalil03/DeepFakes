import { useState, useCallback } from "react";
import Header from "@/components/Header";
import ImageUpload from "@/components/ImageUpload";
import AnalyzeButton from "@/components/AnalyzeButton";
import ResultCard, { type AnalysisResult } from "@/components/ResultCard";
import { Client } from "@gradio/client";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
  }, []);

  const handleImageRemove = useCallback(() => {
    setImage(null);
    setImagePreview(null);
    setResult(null);
  }, []);

  const { toast } = useToast();

  const handleAnalyze = useCallback(async () => {
    if (!image) return;
    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error ${response.status}`);
      }

      const parsedData = await response.json();

      if (parsedData.error) {
        throw new Error(parsedData.error);
      }

      const { prediction, confidence, probabilities, classes } = parsedData;

      // Classes array mapping back to probabilies
      const probMap: Record<string, number> = {};
      classes.forEach((c: string, idx: number) => {
        probMap[c] = probabilities[idx];
      });

      const getProb = (key: string) => (probMap[key] || 0) * 100;

      const is_real = prediction === 'Humano (Real)';

      const finalResult: AnalysisResult = {
        isReal: is_real,
        confidence: confidence * 100,
        categories: {
          human: getProb("Humano (Real)"),
          deepfakeGan: getProb("Deepfake (GAN)"),
          dalle3: getProb("DALL-E 3 (ChatGPT)"),
          midjourneyV6: getProb("Midjourney v6"),
          stableDiffusion: getProb("Stable Diffusion"),
        },
      };

      setResult(finalResult);
    } catch (error) {
      console.error("Inference Error:", error);
      toast({
        title: "Erro de Análise",
        description: "Falha ao conectar com a IA (Hugging Face Spaces).",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [image, toast]);

  return (
    <div className="min-h-screen flex flex-col bg-background grid-bg">
      <Header />

      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 gap-8">
        <div className="text-center mb-2">
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-foreground mb-3">
            Detecte <span className="gradient-text">DeepFakes</span> com Precisão
          </h2>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Envie uma imagem para análise forense com inteligência artificial. Identifique manipulações e atribua a origem do conteúdo.
          </p>
        </div>

        <ImageUpload
          image={image}
          imagePreview={imagePreview}
          onImageSelect={handleImageSelect}
          onImageRemove={handleImageRemove}
          isAnalyzing={isAnalyzing}
        />

        <AnalyzeButton
          onClick={handleAnalyze}
          isAnalyzing={isAnalyzing}
          disabled={!image}
        />

        {result && <ResultCard result={result} />}
      </main>

      <footer className="py-4 text-center">
        <p className="text-xs font-mono text-muted-foreground">
          Veritas AI — Análise Forense Digital &amp; AI Attribution
        </p>
      </footer>
    </div>
  );
};

export default Index;
