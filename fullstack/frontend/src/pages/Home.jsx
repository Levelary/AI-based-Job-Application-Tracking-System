import React from 'react'
import Header from '../components/header';
import Leaderboard from '../components/leaderboard';
import Applications from '../components/applications';
import JobRoles from '../components/jobRoles';

const Home = () => {
  let role = localStorage.getItem('role');
  let id = localStorage.getItem('id');
  return (
    <div>
        <Header />

        {role === 'recruiter' && (
          <>
            <Leaderboard/>
            <Applications/>
          </>
        )}
        {role === 'applicant' && (
          <JobRoles/>
        )}
        
    </div>
  )
}

export default Home;