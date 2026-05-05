"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import { TrendCard } from "@/components/Dashboard/TrendCard";
import { PredictionSection } from "@/components/Dashboard/PredictionSection";
import { motion } from "framer-motion";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";

const data = [
  { name: "00:00", value: 400 },
  { name: "04:00", value: 300 },
  { name: "08:00", value: 600 },
  { name: "12:00", value: 800 },
  { name: "16:00", value: 500 },
  { name: "20:00", value: 900 },
  { name: "23:59", value: 1100 },
];

const mockTrends = [
  { name: "Large Language Models", score: 98.4, velocity: 12.5, sentiment: "Positive" as const, keywords: ["AI", "OpenAI", "DeepLearning"] },
  { name: "Sustainable Aviation Fuel", score: 85.2, velocity: 8.2, sentiment: "Positive" as const, keywords: ["GreenTech", "Aviation", "ESG"] },
  { name: "Central Bank Digital Currencies", score: 76.8, velocity: -2.1, sentiment: "Neutral" as const, keywords: ["Finance", "Crypto", "Policy"] },
  { name: "Autonomous Vehicle Ethics", score: 92.1, velocity: 15.4, sentiment: "Negative" as const, keywords: ["Safety", "Auto", "Ethics"] },
];

export default function Home() {
  return (
    <DashboardLayout>
      <div className="space-y-12">
        {/* Welcome Section */}
        <section>
          <motion.h1 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Insights Overview
          </motion.h1>
          <p className="text-gray-500">Real-time analysis of global news streams and emerging trends.</p>
        </section>

        {/* Hero Chart */}
        <section className="grid grid-cols-3 gap-8">
          <div className="col-span-2 p-8 rounded-3xl border border-white/10 bg-white/5 backdrop-blur-3xl h-[400px]">
            <div className="flex justify-between items-center mb-8">
              <h3 className="font-semibold text-lg text-gray-300">Global Activity Spike</h3>
              <div className="flex gap-4">
                <span className="flex items-center gap-2 text-xs text-purple-400 font-bold">
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                  LIVE UPDATES
                </span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height="80%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Area type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={3} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          
          <div className="p-8 rounded-3xl border border-white/10 bg-gradient-to-br from-purple-500/10 to-blue-500/10 flex flex-col justify-between">
            <div>
              <h3 className="font-semibold text-lg text-gray-300 mb-4">Total Articles</h3>
              <p className="text-5xl font-bold">1.2M+</p>
              <span className="text-green-400 text-sm font-bold tracking-wide">↑ 14.2% VS LAST WEEK</span>
            </div>
            <div className="space-y-4">
              <p className="text-xs text-gray-500 font-bold uppercase tracking-widest">Active Sources</p>
              <div className="flex -space-x-3">
                {[1,2,3,4,5].map(i => (
                  <div key={i} className="w-10 h-10 rounded-full border-2 border-black bg-gray-800" />
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Trends Grid */}
        <section>
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-2xl font-bold tracking-tight">Trending Now</h3>
            <button className="text-sm font-bold text-purple-400 hover:text-purple-300 transition-colors uppercase tracking-widest">View All Trends →</button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {mockTrends.map((trend, index) => (
              <TrendCard key={index} {...trend} />
            ))}
          </div>
        </section>

        {/* Predicted Future Topics */}
        <PredictionSection />
      </div>
    </DashboardLayout>
  );
}
