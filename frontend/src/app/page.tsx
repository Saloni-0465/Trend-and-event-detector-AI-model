"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import { TrendCard } from "@/components/Dashboard/TrendCard";
import { PredictionSection } from "@/components/Dashboard/PredictionSection";
import dashboardData from "@/data/dashboard.json";
import { motion } from "framer-motion";
import { Tooltip, ResponsiveContainer, AreaChart, Area, XAxis, YAxis } from "recharts";
import { useEffect, useState } from "react";

const trends = dashboardData.trends.map((trend) => ({
  ...trend,
  sentiment: trend.sentiment as "Positive" | "Neutral" | "Negative",
}));

const predictions = dashboardData.predictions.map((prediction) => ({
  ...prediction,
  direction: prediction.direction as "rising" | "breakout" | "cooling",
}));

const weeklyChange = dashboardData.stats.weekly_change_pct;

export default function Home() {
  const [chartReady, setChartReady] = useState(false);

  useEffect(() => {
    const id = window.requestAnimationFrame(() => setChartReady(true));
    return () => window.cancelAnimationFrame(id);
  }, []);

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
            {chartReady && (
              <ResponsiveContainer width="100%" height="80%" minWidth={0} minHeight={300}>
                <AreaChart data={dashboardData.activity}>
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
                  <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} width={36} />
                  <Area type="monotone" dataKey="value" stroke="#8b5cf6" strokeWidth={3} fillOpacity={1} fill="url(#colorValue)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
          
          <div className="p-8 rounded-3xl border border-white/10 bg-gradient-to-br from-purple-500/10 to-blue-500/10 flex flex-col justify-between">
            <div>
              <h3 className="font-semibold text-lg text-gray-300 mb-4">Total Articles</h3>
              <p className="text-5xl font-bold">{dashboardData.stats.total_articles_label}</p>
              <span className={`text-sm font-bold tracking-wide ${weeklyChange >= 0 ? "text-green-400" : "text-red-400"}`}>
                {weeklyChange >= 0 ? "↑" : "↓"} {Math.abs(weeklyChange).toFixed(1)}% VS LAST WEEK
              </span>
            </div>
            <div className="space-y-4">
              <p className="text-xs text-gray-500 font-bold uppercase tracking-widest">Active Categories</p>
              <div className="flex -space-x-3">
                {Array.from({ length: Math.min(5, dashboardData.stats.active_sources) }).map((_, i) => (
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
            {trends.map((trend, index) => (
              <TrendCard key={index} {...trend} />
            ))}
          </div>
        </section>

        {/* Predicted Future Topics */}
        <PredictionSection predictions={predictions} />
      </div>
    </DashboardLayout>
  );
}
