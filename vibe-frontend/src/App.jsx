import React, { useState, useRef } from 'react';
import { ChevronLeft, ChevronRight, Upload, Search, ExternalLink, X, Sparkles, Loader2 } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000"

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [topK, setTopK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  const handleSearch = async () => {
    if (!query.trim() && !imageFile) return;
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      let data;

      if (imageFile && !query.trim()) {
        // Image-only → /recommend-from-image
        const formData = new FormData();
        formData.append("file", imageFile);
        const res = await fetch(`${API_BASE}/recommend-from-image?k=${topK}`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Image recommendation failed.");
        }
        data = await res.json();
      } else {
        // Text prompt → /recommend
        const res = await fetch(`${API_BASE}/recommend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: query.trim(), k: topK }),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Recommendation failed.");
        }
        data = await res.json();
      }

      setResults(data);
      setCurrentIndex(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSearch();
  };

  const nextCard = () => setCurrentIndex((prev) => (prev + 1) % results.length);
  const prevCard = () => setCurrentIndex((prev) => (prev - 1 + results.length) % results.length);

  const current = results[currentIndex];

  return (
    <div className="min-h-screen bg-[#F9F7F2] text-[#2D2D2D] font-sans selection:bg-orange-100">
      
      {/* HEADER */}
      <nav className="p-6 flex justify-between items-center bg-transparent max-w-6xl mx-auto">
        <div className="flex items-center gap-2">
          <Sparkles className="text-orange-500" size={24}/>
          <span className="text-xl font-semibold tracking-tight">VibeEngine</span>
        </div>
        <div className="text-[10px] text-gray-400 font-mono tracking-widest uppercase">
          GT Hacklytics 2026
        </div>
      </nav>

      <main className="max-w-3xl mx-auto pt-20 pb-12 px-6 flex flex-col items-center">
        
        {/* HERO */}
        <h1 className="text-5xl font-bold tracking-tight mb-4 text-center">
          What vibe are you <br/> feeling today?
        </h1>
        <p className="text-gray-500 mb-12 text-center text-lg">
          Music discovery, reimagined
        </p>

        {/* INPUT BOX */}
        <section className="w-full bg-white border border-gray-200 rounded-3xl p-4 shadow-sm focus-within:shadow-md transition-shadow">
          <div className="relative">
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full h-32 bg-transparent p-4 text-lg outline-none resize-none placeholder:text-gray-300"
              placeholder="Describe a mood, a memory, or a lyric..."
            />
            {imagePreview && (
              <div className="absolute bottom-4 left-4 group">
                <img src={imagePreview} className="w-16 h-16 object-cover rounded-xl border border-gray-100 shadow-sm" alt="preview"/>
                <button onClick={clearImage} className="absolute -top-2 -right-2 bg-gray-800 text-white rounded-full p-1">
                  <X size={10}/>
                </button>
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mt-2 border-t border-gray-50 pt-4 px-2">
            <div className="flex items-center gap-4">
              <button 
                onClick={() => fileInputRef.current.click()}
                className="text-gray-400 hover:text-orange-500 transition-colors"
                title="Attach Vibe Image"
              >
                <Upload size={20}/>
                <input type="file" ref={fileInputRef} onChange={handleImageUpload} className="hidden" accept="image/jpeg,image/png" />
              </button>
              <span className="text-xs text-gray-300">Translate an image into song recs!</span>

              {/* Top K selector */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Results</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="text-xs text-gray-500 bg-gray-50 border border-gray-100 rounded-lg px-2 py-1 outline-none cursor-pointer"
                >
                  {[5, 10, 15, 20].map(n => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
            </div>

            <button 
              onClick={handleSearch}
              disabled={loading || (!query.trim() && !imageFile)}
              className="bg-[#2D2D2D] text-white p-3 rounded-2xl hover:bg-black transition-all shadow-lg flex items-center justify-center aspect-square disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? <Loader2 size={22} className="animate-spin"/> : <Search size={22}/>}
            </button>
          </div>
        </section>

        {/* ERROR STATE */}
        {error && (
          <div className="w-full mt-6 p-4 bg-red-50 border border-red-100 rounded-2xl text-center">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* LOADING STATE */}
        {loading && (
          <div className="mt-16 flex flex-col items-center gap-3 text-gray-300">
            <Loader2 size={32} className="animate-spin text-orange-400"/>
            <p className="text-xs font-mono uppercase tracking-widest">Searching the vibe...</p>
          </div>
        )}

        {/* RESULTS CAROUSEL */}
        {!loading && results.length > 0 && (
          <div className="w-full mt-12 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Enriched query summary */}
            {current?.enriched && (
              <div className="text-center mb-6">
                <p className="text-[10px] text-black-400 font-mono uppercase tracking-widest">
                  🧠 {current.enriched.rewritten_prompt}
                </p>
                <p className="text-[10px] text-black-300 mt-1">
                  {current.enriched.moods.join(" · ")} · ⚡ {current.enriched.energy}
                </p>
              </div>
            )}

            <div className="flex items-center justify-center gap-4">

              <button
                onClick={prevCard}
                className="p-3 bg-white hover:bg-gray-50 border border-gray-100 rounded-2xl text-gray-400 hover:text-black shadow-md transition-all flex-shrink-0"
              >
                <ChevronLeft size={22}/>
              </button>

              {/* Card */}
              <div className="w-[380px] bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-xl flex flex-col items-center gap-6">
                
                {/* Album Art */}
                <div className="w-72 h-72 flex-shrink-0 relative group">
                  {current.album_art ? (
                    <img 
                      src={current.album_art}
                      className="w-full h-full object-cover rounded-3xl shadow-md transition-transform duration-300 group-hover:scale-[1.02]" 
                      alt="Album Art"
                    />
                  ) : (
                    <div className="w-full h-full rounded-3xl bg-gray-100 flex items-center justify-center">
                      <Sparkles size={40} className="text-gray-300"/>
                    </div>
                  )}
                </div>

                {/* Content */}
                <div className="w-full flex flex-col gap-4 text-center">
                  <div>
                    <h2 className="text-2xl font-bold text-[#2D2D2D] leading-tight">
                      {current.title}
                    </h2>
                    <p className="text-sm text-gray-400 font-medium uppercase tracking-wide">
                      {current.artist}{current.year ? ` · ${current.year}` : ""}
                    </p>
                  </div>

                  {/* Score pills */}
                  <div className="flex justify-center gap-4 text-[10px] font-mono text-gray-300 uppercase tracking-widest">
                    <span>Semantic Match {(current.semantic_score * 100).toFixed(0)}%</span>
                    <span>·</span>
                    <span>Final Score {(current.final_score * 100).toFixed(0)}%</span>
                    {current.views && (
                      <span>· {(current.views / 1_000_000).toFixed(1)}M views</span>
                    )}
                  </div>

                  {/* Lyric Snippet */}
                  <div className="bg-[#F9F7F2] p-4 rounded-2xl border border-gray-50">
                    <p className="text-[10px] font-black text-orange-400 uppercase tracking-widest mb-1 flex justify-center items-center gap-1">
                      <Sparkles size={12}/> Lyric Snippet
                    </p>
                    <p className="text-xs text-gray-600 leading-relaxed italic">
                      "{current.lyric_snippet}"
                    </p>
                  </div>

                  {/* Explainable AI */}
                  {current.explanation && (
                  <div className="bg-[#F9F7F2] p-4 rounded-2xl border border-gray-50">
                    <p className="text-[10px] font-black text-orange-400 uppercase tracking-widest mb-1 flex justify-center items-center gap-1">
                      <Sparkles size={12}/> Why this song?
                    </p>
                    <p className="text-xs text-gray-600 leading-relaxed italic">
                      "{current.explanation}"
                    </p>
                  </div>
                )}
                  {/* Spotify */}
                  {current.spotify_url ? (
                    <a 
                      href={current.spotify_url} 
                      target="_blank"
                      rel="noreferrer"
                      className="w-full inline-flex items-center justify-center gap-2 bg-[#2D2D2D] text-white py-3 rounded-2xl font-bold hover:bg-black transition-all text-sm"
                    >
                      Spotify <ExternalLink size={16}/>
                    </a>
                  ) : (
                    <div className="w-full inline-flex items-center justify-center bg-gray-100 text-gray-300 py-3 rounded-2xl text-sm cursor-not-allowed">
                      No Spotify Link
                    </div>
                  )}
                </div>
              </div>

              <button
                onClick={nextCard}
                className="p-3 bg-white hover:bg-gray-50 border border-gray-100 rounded-2xl text-gray-400 hover:text-black shadow-md transition-all flex-shrink-0"
              >
                <ChevronRight size={22}/>
              </button>
            </div>
            
            <p className="text-center mt-6 text-[10px] text-gray-300 font-mono uppercase tracking-widest">
              {currentIndex + 1} / {results.length}
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
