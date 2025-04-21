import React from 'react';
import styles from './Auth.module.css';
import { Link, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/auth/login', {
        email,
        password,
      });

      if(response.data.success) {
        setMessage('Login successful! Redirecting to home page...');

        localStorage.setItem('isLoggedIn', true);
        localStorage.setItem('role', response.data.role); // Storing user data in local storage
        localStorage.setItem('id', response.data.id); 

        setTimeout(() => {
          navigate('/'); // Redirect to home page after success
        }, 2000); // Wait for 2 seconds before redirect
      } 
      else {
        setMessage(response.data.message || 'Login Failed!');
      }

    }
    catch (error) { 
      console.error('Login Error:', error);

      if(error.response && error.response.data) 
        setMessage(error.response.data.message || 'Login request failed!');
      
      else
        setMessage('Login request failed!');
    }
  }
    

  return (
    <div className={styles.container}>
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <div>
          <input type='email' placeholder='Email' value={email} onChange={(e) => setEmail(e.target.value)} required/>
        </div>
        <br/>
        <div>
          <input type='password' placeholder='Password' value={password} onChange={(e) => setPassword(e.target.value)} required/>
        </div>
        <button type='submit'>Login</button>
      </form>
      {message && <p>{message}</p>} {/* if message not empty, display it */}
      <br/>
      <Link to='/signup'>A new user ?</Link>

    </div>
  )
}

export default Login