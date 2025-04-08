import { useState } from 'react'
import Header from './header'
import './App.css'

function App() {

  return (
    <div>
      <Header />
      <div className="header">
        <h3 className="title">Gift Bot</h3>
      </div>
      <div className="chatBody">
        <Icon className="chatbotIcon"/>
        <div className="botMessage">
          <p className="messageText">Hello there! Do you need my assistance with selecting a gift for someone</p>
        </div>
        <div className="userMessage">
          <p className="messageText">Yes please</p>
        </div>
      </div>
      <div className="footer">
        <form className="inputForm">
          <input type="text" className="inputBox" placeholder='Enter your response here...'/>
          <button className="inputBotton"></button>
        </form>

      </div>
    </div>
  )
}

export default App
