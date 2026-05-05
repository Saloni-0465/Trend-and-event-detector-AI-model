"use client";
import { motion } from "framer-motion";
import { TrendingUp, MessageCircle, BarChart2 } from "lucide-react";

interface TrendCardProps {
  name: string;
  score: number;
  velocity: number;
  sentiment: "Positive" | "Neutral" | "Negative";
  keywords: string[];
}

export const TrendCard = ({ name, score, velocity, sentiment, keywords }: TrendCardProps) => {
  const sentimentColor = 
    sentiment === "Positive" ? "text-green-400" : 
    sentiment === "Negative" ? "text-red-400" : "text-gray-400";

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl hover:border-white/20 transition-all cursor-pointer"
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-semibold text-white truncate pr-2">{name}</h3>
        <span className={`px-3 py-1 rounded-full text-xs font-medium bg-white/10 ${sentimentColor}`}>
          {sentiment}
        </span>
      </div>
      
      <div className="flex gap-4 mb-6">
        <div className="flex items-center gap-1.5 text-blue-400">
          <BarChart2 size={16} />
          <span className="text-sm font-bold">{score.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-1.5 text-purple-400">
          <TrendingUp size={16} />
          <span className="text-sm font-bold">+{velocity}%</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {keywords.slice(0, 3).map((kw) => (
          <span key={kw} className="px-2 py-0.5 rounded-md bg-white/5 text-[10px] text-gray-400 uppercase tracking-wider">
            #{kw}
          </span>
        ))}
      </div>
    </motion.div>
  );
};
