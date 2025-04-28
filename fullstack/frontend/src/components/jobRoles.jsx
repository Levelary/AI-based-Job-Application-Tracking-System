import React, { useState, useEffect } from 'react';
import axios from 'axios';

const jobRoles = () => {
  const [jobRoles, setjobRoles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:5000/jobRoles')
    .then((res) => {
      setjobRoles(res.data);
      setLoading(False);
    })
    .catch((err) => {
      console.error('Error fetching job roles: ', err);
      setLoading(False);
    });
  }, []);


  const handleApply = (jobId) => {
    const userId = localStorage.getItem('id');
    const userRole = localStorage.getItem('role');
    if(userRole === 'applicant') {
      axios.post(`http://localhost:5000/jobRoles/${jobId}/apply`, { userId })
      .then((res) => {
        alert(res.data.message);
      })
      .catch((err) => {
        console.error('Error applying for job: ', err);
      });
    } else {
      alert('You are not authorized to apply for jobs!');
    }
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>job Eligibility Leaderboard</h2>
      { loading ? ( <p> Loading job roles... </p> ) 
      : (
          <div>
            {jobRoles.map((job, index) => (
              <div key={job.ID} style={styles.jobCard}>
                <p><strong>Company:</strong> {job.companyName}</p>
                <p><strong>Role:</strong> {job.jobRole}</p>
                <button onClick={() => handleApply(job.ID)}>Apply</button>
              </div>
            ))}
          </div>
        )
      }
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    fontFamily: 'Arial, sans-serif',
    textAlign: 'center'
  },
  title: {
    fontSize: '24px',
    marginBottom: '20px'
  },
  table: {
    margin: '0 auto',
    borderCollapse: 'collapse',
    width: '80%'
  },
  headerCell: {
    backgroundColor: '#007bff',
    color: 'white',
    padding: '10px',
    border: '1px solid #ccc'
  },
  cell: {
    padding: '10px',
    border: '1px solid #ccc',
    textAlign: 'center'
  }
};



export default jobRoles;