"use client";
import { DashboardLayout } from "@/components/Dashboard/DashboardLayout";
import dashboardData from "@/data/dashboard.json";
import { motion } from "framer-motion";
import { Database } from "lucide-react";

interface Source {
  id: number;
  name: string;
  type: "dataset";
  url: string;
  status: "active";
  lastFetch: string;
  articlesTotal: number;
  coveragePct: number;
}

const initialSources: Source[] = dashboardData.sources.map((source, index) => ({
  id: index + 1,
  name: source.name,
  type: "dataset",
  url: `research/data/raw/${dashboardData.meta.dataset}#${source.name.toLowerCase().replaceAll(" ", "-")}`,
  status: "active",
  lastFetch: source.lastFetch,
  articlesTotal: source.articlesTotal,
  coveragePct: source.coveragePct,
}));

const statusConfig = {
  active: { color: "text-green-400", label: "Loaded", dot: "bg-green-500" },
};

export default function SourcesPage() {
  const sources = initialSources;
  const activeCount = sources.filter((s) => s.status === "active").length;
  const totalArticles = sources.reduce((sum, s) => sum + s.articlesTotal, 0);

  return (
    <DashboardLayout>
      <div className="space-y-10">
        <section>
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent"
          >
            Data Sources
          </motion.h1>
          <p className="text-gray-500">Dataset category coverage used by the TF-IDF, LDA, embedding, and clustering pipeline.</p>
        </section>

        {/* Stats */}
        <section className="grid grid-cols-3 gap-6">
          {[
            { label: "Total Sources", value: sources.length.toString(), color: "from-purple-500/20 to-purple-500/5" },
            { label: "Loaded", value: activeCount.toString(), color: "from-green-500/20 to-green-500/5" },
            { label: "Articles Ingested", value: `${(totalArticles / 1000).toFixed(0)}K`, color: "from-blue-500/20 to-blue-500/5" },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className={`p-6 rounded-2xl border border-white/10 bg-gradient-to-br ${stat.color}`}
            >
              <p className="text-xs text-gray-500 uppercase tracking-widest font-bold mb-2">{stat.label}</p>
              <p className="text-3xl font-bold">{stat.value}</p>
            </motion.div>
          ))}
        </section>

        {/* Source Table */}
        <section>
          <h2 className="text-xl font-bold mb-6 tracking-tight">All Sources</h2>
          <div className="rounded-2xl border border-white/10 overflow-hidden">
            {/* Header */}
            <div className="grid grid-cols-12 gap-4 px-6 py-3 bg-white/5 text-xs text-gray-500 uppercase tracking-widest font-bold">
              <div className="col-span-4">Category</div>
              <div className="col-span-2">Type</div>
              <div className="col-span-2">Status</div>
              <div className="col-span-2">Coverage</div>
              <div className="col-span-2 text-right">Articles</div>
            </div>

            {/* Rows */}
            {sources.map((source, i) => {
              const sCfg = statusConfig[source.status];

              return (
                <motion.div
                  key={source.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.06 }}
                  className="grid grid-cols-12 gap-4 px-6 py-4 border-t border-white/5 hover:bg-white/[0.03] transition-all items-center"
                >
                  <div className="col-span-4">
                    <p className="font-medium text-sm text-white">{source.name}</p>
                    <p className="text-xs text-gray-600 truncate">Latest article: {source.lastFetch}</p>
                  </div>
                  <div className="col-span-2 flex items-center gap-2 text-sm text-gray-400">
                    <Database size={14} />
                    <span className="capitalize">{source.type}</span>
                  </div>
                  <div className="col-span-2 flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${sCfg.dot} ${source.status === "active" ? "animate-pulse" : ""}`} />
                    <span className={`text-sm font-medium ${sCfg.color}`}>{sCfg.label}</span>
                  </div>
                  <div className="col-span-2 text-sm text-gray-500">{source.coveragePct.toFixed(1)}%</div>
                  <div className="col-span-2 text-right text-sm text-gray-300">{source.articlesTotal.toLocaleString()}</div>
                </motion.div>
              );
            })}
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
