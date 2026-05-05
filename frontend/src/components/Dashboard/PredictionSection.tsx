"use client";
import { motion } from "framer-motion";
import { Brain, TrendingUp, TrendingDown, Rocket, Clock, Sparkles, ArrowRight } from "lucide-react";

interface Prediction {
  id: number;
  topic: string;
  predicted_score: number;
  velocity_forecast: number;
  current_mentions: number;
  predicted_mentions: number;
  confidence: number;
  horizon: string;
  drivers: string[];
  category: string;
  direction: "rising" | "breakout" | "cooling";
}

const directionConfig = {
  breakout: {
    icon: Rocket,
    color: "text-purple-400",
    bg: "from-purple-500/20 to-purple-500/5",
    border: "border-purple-500/30",
    label: "Breakout",
    glow: "shadow-purple-500/10",
  },
  rising: {
    icon: TrendingUp,
    color: "text-green-400",
    bg: "from-green-500/20 to-green-500/5",
    border: "border-green-500/30",
    label: "Rising",
    glow: "shadow-green-500/10",
  },
  cooling: {
    icon: TrendingDown,
    color: "text-orange-400",
    bg: "from-orange-500/20 to-orange-500/5",
    border: "border-orange-500/30",
    label: "Cooling",
    glow: "shadow-orange-500/10",
  },
};

const mockPredictions: Prediction[] = [
  {
    id: 1,
    topic: "Quantum Error Correction",
    predicted_score: 94.2,
    velocity_forecast: 28.5,
    current_mentions: 120,
    predicted_mentions: 890,
    confidence: 0.91,
    horizon: "48h",
    drivers: ["Google Willow", "logical qubit", "error rate"],
    category: "Technology",
    direction: "breakout",
  },
  {
    id: 2,
    topic: "US-China Semiconductor Tariffs",
    predicted_score: 88.7,
    velocity_forecast: 19.3,
    current_mentions: 340,
    predicted_mentions: 1200,
    confidence: 0.87,
    horizon: "24h",
    drivers: ["CHIPS Act", "export controls", "TSMC"],
    category: "Geopolitics",
    direction: "rising",
  },
  {
    id: 3,
    topic: "Weight Loss Drug Patent Wars",
    predicted_score: 82.1,
    velocity_forecast: 14.8,
    current_mentions: 210,
    predicted_mentions: 720,
    confidence: 0.83,
    horizon: "48h",
    drivers: ["Ozempic", "GLP-1", "FDA approval"],
    category: "Health",
    direction: "rising",
  },
  {
    id: 4,
    topic: "Arctic Shipping Route Expansion",
    predicted_score: 71.5,
    velocity_forecast: 22.1,
    current_mentions: 45,
    predicted_mentions: 380,
    confidence: 0.76,
    horizon: "7d",
    drivers: ["ice melt", "Northern Sea Route", "cargo volume"],
    category: "Climate",
    direction: "breakout",
  },
  {
    id: 5,
    topic: "Creator Economy Regulation",
    predicted_score: 67.3,
    velocity_forecast: 9.4,
    current_mentions: 180,
    predicted_mentions: 410,
    confidence: 0.72,
    horizon: "7d",
    drivers: ["influencer tax", "FTC guidelines", "disclosure"],
    category: "Business",
    direction: "rising",
  },
  {
    id: 6,
    topic: "Satellite Internet Coverage Gap",
    predicted_score: 58.9,
    velocity_forecast: -4.2,
    current_mentions: 290,
    predicted_mentions: 210,
    confidence: 0.68,
    horizon: "48h",
    drivers: ["Starlink", "rural broadband", "spectrum"],
    category: "Technology",
    direction: "cooling",
  },
];

function ScoreBar({ score }: { score: number }) {
  const width = Math.min(100, Math.max(0, score));
  const hue = score > 80 ? 270 : score > 60 ? 142 : 38; // purple > green > orange
  return (
    <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${width}%` }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="h-full rounded-full"
        style={{ background: `hsl(${hue}, 70%, 55%)` }}
      />
    </div>
  );
}

export const PredictionSection = () => {
  return (
    <section className="space-y-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/20 flex items-center justify-center">
            <Brain size={20} className="text-purple-400" />
          </div>
          <div>
            <h3 className="text-2xl font-bold tracking-tight flex items-center gap-2">
              Predicted Future Topics
              <Sparkles size={18} className="text-purple-400 animate-pulse" />
            </h3>
            <p className="text-xs text-gray-500">AI-forecasted topics likely to trend in the next 24h–7d</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-[10px] text-gray-400 uppercase tracking-widest font-bold">
          <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
          Model Active
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {mockPredictions.map((pred, i) => {
          const cfg = directionConfig[pred.direction];
          const DirIcon = cfg.icon;
          return (
            <motion.div
              key={pred.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              whileHover={{ scale: 1.015, y: -2 }}
              className={`p-5 rounded-2xl border border-white/10 bg-gradient-to-br ${cfg.bg} backdrop-blur-xl cursor-pointer group hover:border-white/20 hover:shadow-xl ${cfg.glow} transition-all`}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${cfg.color} bg-white/5`}>
                      {cfg.label}
                    </span>
                    <span className="text-[10px] text-gray-600 flex items-center gap-1">
                      <Clock size={9} /> {pred.horizon}
                    </span>
                  </div>
                  <h4 className="font-semibold text-white text-sm leading-tight group-hover:text-purple-200 transition-colors">
                    {pred.topic}
                  </h4>
                </div>
                <div className="text-right ml-2 shrink-0">
                  <p className="text-2xl font-bold text-white">{pred.predicted_score.toFixed(0)}</p>
                  <p className="text-[9px] text-gray-500 uppercase tracking-wider">Score</p>
                </div>
              </div>

              {/* Score Bar */}
              <ScoreBar score={pred.predicted_score} />

              {/* Stats Row */}
              <div className="flex items-center justify-between mt-3 text-xs">
                <div className="flex items-center gap-1">
                  <DirIcon size={12} className={cfg.color} />
                  <span className={cfg.color}>
                    {pred.velocity_forecast > 0 ? "+" : ""}
                    {pred.velocity_forecast}% vel
                  </span>
                </div>
                <span className="text-gray-500">
                  {pred.current_mentions} → <span className="text-white font-medium">{pred.predicted_mentions}</span> mentions
                </span>
              </div>

              {/* Confidence */}
              <div className="flex items-center justify-between mt-2 text-xs">
                <span className="text-gray-600">{pred.category}</span>
                <span className="text-gray-500">
                  Confidence: <span className="text-white font-bold">{(pred.confidence * 100).toFixed(0)}%</span>
                </span>
              </div>

              {/* Drivers */}
              <div className="flex flex-wrap gap-1.5 mt-3">
                {pred.drivers.map((d) => (
                  <span
                    key={d}
                    className="px-2 py-0.5 rounded-md bg-white/5 text-[9px] text-gray-400 uppercase tracking-wider"
                  >
                    {d}
                  </span>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
};
