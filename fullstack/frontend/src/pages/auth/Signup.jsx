import React from 'react';
import styles from './Auth.module.css';
import { Link, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';

const Signup = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [verifyPassword, setVerifyPassword] = useState('');
  const [message, setMessage] = useState('');
  const [role, setRole] = useState('applicant');
  const navigate = useNavigate();
  
  const handleSignup = async (e) => {
    e.preventDefault();

    if(password !== verifyPassword) {
      setMessage('Passwords do not match!');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/auth/signup', {
        email,
        password,
        role,
      });

      if(response.data.success) {
        setMessage('Signup successful! Redirecting to login page...');

        setTimeout(() => {
          navigate('/login'); // Redirect to login page after success
        }, 2000); // Wait for 2 seconds before redirect
      } 
      else {
        setMessage(response.data.error || 'Login Failed!');
      }

    }
    catch (error) { 
      console.error('Error:', error);
      
      
      if(error.response && error.response.data) 
        setMessage(error.response.data.error || 'Signup request failed!');
      
      else
        setMessage('Signup request failed!');
    }
  }
  return (
    <div className={styles.container} style={{ height: '50vh' }}>
    <h2>Signup</h2>
    <form onSubmit={handleSignup}>
        <div>
          <input type='email' placeholder='Email' value={email} onChange={(e) => setEmail(e.target.value)} required/>
        </div>
        <br/>
        <div>
          <input type='password' placeholder='Password' value={password} onChange={(e) => setPassword(e.target.value)} required/>
        </div>
        <div>
          <input type='password' placeholder='Re-enter Password' value={verifyPassword} onChange={(e) => setVerifyPassword(e.target.value)} required/>
        </div>
        <div>
          <label htmlFor='role'>Select your role:</label>
          <select id='role' value={role} onChange={(e) => setRole(e.target.value)} required>
            <option value='recruiter'>Recruiter</option>
            <option value='applicant'>Applicant</option>
          </select>
        </div>
        <button type='submit'>Signup</button>
      </form>
      {message && <p>{message}</p>} {/* if message not empty, display it */}
      <br/>
    <Link to='/login'>Already have an account ?</Link>
  </div>
  )
}

export default Signup;

