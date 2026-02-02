import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Predictions from "./pages/Predictions";
import Search from "./pages/Search";
import NextSeason from "./pages/NextSeason";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predictions" element={<Predictions />} />
        <Route path="/search" element={<Search />} />
        <Route path="/next-season" element={<NextSeason />} />
      </Routes>
    </BrowserRouter>
  );
}
