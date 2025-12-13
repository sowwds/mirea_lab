const request = require('supertest');
const app = require('../app'); // Import the Express app

describe('App Endpoints', () => {

  // Test 1: Check if the root endpoint is running
  it('should return 200 OK for the root endpoint', async () => {
    const response = await request(app).get('/');
    expect(response.statusCode).toBe(200);
    expect(response.text).toBe('Backend is running!');
  });

  // Test 2: Check if a protected route is actually protected
  it('should return 401 Unauthorized for a protected route without a token', async () => {
    const response = await request(app).get('/api/defects');
    expect(response.statusCode).toBe(401);
  });

  // Test 3: Check the registration and login flow
  describe('Authentication Flow', () => {
    let token = '';
    // Use a random user for each test run to avoid conflicts
    const randomEmail = `testuser_${Date.now()}@example.com`;
    const password = 'password123';

    it('should register a new user successfully', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: randomEmail,
          password: password,
        });
      expect(response.statusCode).toBe(201);
      expect(response.body).toHaveProperty('message', 'User created successfully');
    });

    it('should log in the new user and return a token', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: randomEmail,
          password: password,
        });
      expect(response.statusCode).toBe(200);
      expect(response.body).toHaveProperty('token');
      token = response.body.token; // Save token for the next test
    });

    it('should access a protected route with a valid token', async () => {
      const response = await request(app)
        .get('/api/defects')
        .set('Authorization', `Bearer ${token}`);
      expect(response.statusCode).toBe(200);
    });
  });
});
