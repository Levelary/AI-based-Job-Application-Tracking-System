import React, {useState, useEffect} from "react"
import { Routes, Route } from 'react-router-dom'
import './App.css'
import Login from './pages/auth/Login'
import Signup from './pages/auth/Signup'
import Home from './pages/Home'
function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(localStorage.getItem('isLoggedIn') === 'true');

  useEffect(() => {
    const handleChange = () => { // checks if user logs in another browser
      setIsLoggedIn(localStorage.getItem('isLoggedIn') === 'true');
    }
    window.addEventListener('storage', handleChange);
    return () => {
      window.removeEventListener('storage', handleChange);
    }
  }
  , []);

  return (
    <Routes>
      <Route path="/login" element={<Login/>}/>
      <Route path="/signup" element={<Signup/>}/>
      <Route path="/" element={<Login setIsLoggedIn={setIsLoggedIn}/>}/>
    </Routes>
  )
}

export default App
