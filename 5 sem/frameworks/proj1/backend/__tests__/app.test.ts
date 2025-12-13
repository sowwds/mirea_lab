const request = require('supertest');
const app = require('../app'); // Import the Express app
const db = require('../db'); // Import db for cleanup

describe('API Endpoints', () => {
  let token = '';
  let userId = '';
  let assigneeId = '';
  const testUserEmail = `testuser_${Date.now()}@example.com`;
  const assigneeEmail = `assignee_${Date.now()}@example.com`;
  const password = 'password123';

  // Use beforeAll to set up the state needed for all tests
  beforeAll(async () => {
    // Register the main test user
    const userRes = await request(app)
      .post('/api/auth/register')
      .send({ email: testUserEmail, password: password, role: 'ENGINEER' });
    userId = userRes.body.userId;

    // Register the assignee user
    const assigneeRes = await request(app)
      .post('/api/auth/register')
      .send({ email: assigneeEmail, password: password, role: 'ENGINEER' });
    assigneeId = assigneeRes.body.userId;

    // Log in as the main test user to get a token
    const loginRes = await request(app)
      .post('/api/auth/login')
      .send({ email: testUserEmail, password: password });
    token = loginRes.body.token;
  });

  // Clean up created users after all tests are done
  afterAll(async () => {
    await db.query('DELETE FROM "User" WHERE email = $1 OR email = $2', [testUserEmail, assigneeEmail]);
    // Close the pool to prevent Jest from hanging
    await db.pool.end();
  });


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

  // Test 3: Check if a protected route works with a valid token
  it('should return 200 OK for a protected route with a valid token', async () => {
    const response = await request(app)
      .get('/api/defects')
      .set('Authorization', `Bearer ${token}`);
    expect(response.statusCode).toBe(200);
  });
  
  // Test 4: Check if defect creation works
  it('should create a new defect successfully', async () => {
    const response = await request(app)
      .post('/api/defects')
      .set('Authorization', `Bearer ${token}`)
      .send({
        title: 'Test Defect',
        description: 'A test description',
        priority: 'MEDIUM',
        assigneeId: assigneeId
      });
    expect(response.statusCode).toBe(201);
    expect(response.body).toHaveProperty('id');
    expect(response.body.title).toBe('Test Defect');
  });

  // Test 5: Check if role switching works
  it('should allow a user to change their own role', async () => {
    const response = await request(app)
      .put('/api/users/role')
      .set('Authorization', `Bearer ${token}`)
      .send({ role: 'MANAGER' });
    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('token');
    expect(response.body.message).toBe('Role updated to MANAGER');
  });

});
