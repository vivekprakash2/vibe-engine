import React, { useState, useRef } from 'react';
import { ChevronLeft, ChevronRight, Upload, Search, Settings2, Music, ExternalLink, X, Sparkles } from 'lucide-react';

// --- MOCK DATA FOR UI TESTING ---
const MOCK_RECOMMENDATIONS = [
  {
    title: "Child's Play",
    artist: "Drake",
    explanation: "Matches your focus on relationship dynamics and retail therapy, identified via specific semantic markers in the lyrical structure.",
    spotify_url: "https://open.spotify.com",
    album_art: "https://i.scdn.co/image/ab67616d0000b27394185ef3565e3ef5487779f3",
  },
  {
    title: "Dancing Queen",
    artist: "ABBA",
    explanation: "Aligns with the 'upbeat' and 'joyful' sentiment analysis of your prompt.",
    spotify_url: "https://open.spotify.com",
    album_art: "https://i.scdn.co/image/ab67616d0000b27370f612502687a4a93a0279c1",
  }
];

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [imagePreview, setImagePreview] = useState(null);
  const [semanticWeight, setSemanticWeight] = useState(0.7);
  const [topK, setTopK] = useState(5);

  const fileInputRef = useRef(null);

  const handleSearch = () => {
    setResults(MOCK_RECOMMENDATIONS);
    setCurrentIndex(0);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) setImagePreview(URL.createObjectURL(file));
  };

  const nextCard = () => setCurrentIndex((prev) => (prev + 1) % results.length);
  const prevCard = () => setCurrentIndex((prev) => (prev - 1 + results.length) % results.length);

  return (
    <div className="min-h-screen bg-[#F9F7F2] text-[#2D2D2D] font-sans selection:bg-orange-100">
      
      {/* HEADER: Minimalist */}
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
        
        {/* HERO TITLE */}
        <h1 className="text-5xl font-bold tracking-tight mb-4 text-center">
          What vibe are you <br/> feeling today?
        </h1>
        <p className="text-gray-500 mb-12 text-center text-lg">
          Find the song you're feeling
        </p>

        {/* INPUT SECTION: Claude-style Input Box */}
        <section className="w-full bg-white border border-gray-200 rounded-3xl p-4 shadow-sm focus-within:shadow-md transition-shadow">
          <div className="relative">
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full h-32 bg-transparent p-4 text-lg outline-none resize-none placeholder:text-gray-300"
              placeholder="Describe a mood, a memory, or a lyric..."
            />
            {imagePreview && (
              <div className="absolute bottom-4 left-4 group">
                <img src={imagePreview} className="w-16 h-16 object-cover rounded-xl border border-gray-100 shadow-sm" />
                <button onClick={() => setImagePreview(null)} className="absolute -top-2 -right-2 bg-gray-800 text-white rounded-full p-1"><X size={10}/></button>
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mt-2 border-t border-gray-50 pt-4 px-2">
            <div className="flex items-center gap-6">
               <button 
                  onClick={() => fileInputRef.current.click()}
                  className="text-gray-400 hover:text-orange-500 transition-colors"
                  title="Attach Vibe Image"
                >
                  <Upload size={20}/>
                  <input type="file" ref={fileInputRef} onChange={handleImageUpload} className="hidden" accept="image/*" />
               </button>
               <text>Translate an image into song recs!</text>
            </div>

            <button 
              onClick={handleSearch}
              className="bg-[#2D2D2D] text-white p-3 rounded-2xl hover:bg-black transition-all shadow-lg flex items-center justify-center aspect-square"
            >
              <Search size={22}/>
            </button>
          </div>
        </section>
            {/* RESULTS: Square Carousel Card */}
{results.length > 0 && (
  <div className="w-full mt-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
    {/* Flex row: button | card | button */}
    <div className="flex items-center justify-center gap-4">

      {/* Left Button */}
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
          <img 
            src={results[currentIndex].album_art} 
            className="w-full h-full object-cover rounded-3xl shadow-md transition-transform duration-300 group-hover:scale-[1.02]" 
            alt="Album Art"
          />
        </div>

        {/* Content */}
        <div className="w-full flex flex-col gap-4 text-center">
          <div>
            <h2 className="text-2xl font-bold text-[#2D2D2D] leading-tight">
              {results[currentIndex].title}
            </h2>
            <p className="text-sm text-gray-400 font-medium uppercase tracking-wide">
              {results[currentIndex].artist}
            </p>
          </div>
          
          <div className="bg-[#F9F7F2] p-4 rounded-2xl border border-gray-50">
            <p className="text-[10px] font-black text-orange-400 uppercase tracking-widest mb-1 flex justify-center items-center gap-1">
              <Sparkles size={12}/> AI Insight
            </p>
            <p className="text-xs text-gray-600 leading-relaxed italic">
              "{results[currentIndex].explanation}"
            </p>
          </div>

          <a 
            href={results[currentIndex].spotify_url} 
            target="_blank" 
            className="w-full inline-flex items-center justify-center gap-2 bg-[#2D2D2D] text-white py-3 rounded-2xl font-bold hover:bg-black transition-all text-sm"
          >
            Spotify <ExternalLink size={16}/>
          </a>
        </div>
      </div>

      {/* Right Button */}
      <button
        onClick={nextCard}
        className="p-3 bg-white hover:bg-gray-50 border border-gray-100 rounded-2xl text-gray-400 hover:text-black shadow-md transition-all flex-shrink-0"
      >
        <ChevronRight size={22}/>
      </button>
    </div>
    
    {/* Page Indicator */}
    <p className="text-center mt-6 text-[10px] text-gray-300 font-mono uppercase tracking-widest">
      {currentIndex + 1} / {results.length}
    </p>
  </div>
)}
      </main>
    </div>
  );
}