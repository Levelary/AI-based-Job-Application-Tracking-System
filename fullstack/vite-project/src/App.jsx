import React from "react"
import { Routes, Route} from 'react-router-dom'
import './App.css'
import Login from './pages/auth/Login'
import Signup from './pages/auth/Signup'
import Home from './pages/Home'
function App() {

  return (
    <Routes>
      {/* <Route path="/login" element={<Login/>}>
      <Route path="/signup" element={<Signup/>}/> */}
      <Route path="/" element={<Home/>}/>
    </Routes>
  )
}

export default App
