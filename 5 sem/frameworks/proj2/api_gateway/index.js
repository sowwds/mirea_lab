const express = require('express');
const cors = require('cors');
const axios = require('axios');
const CircuitBreaker = require('opossum');
const { rateLimit } = require('express-rate-limit');
const { v4: uuidv4 } = require('uuid');
const jwt = require('jsonwebtoken');

const app = express();
const PORT = process.env.PORT || 8000;
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-key';

const USERS_SERVICE_URL = 'http://service_users:8001';
const ORDERS_SERVICE_URL = 'http://service_orders:8002';

app.use(cors());
app.use(express.json());

const addRequestId = (req, res, next) => {
    req.id = uuidv4();
    next();
};
app.use(addRequestId);

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);

// 3. JWT Authentication Middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer <token>

    if (token == null) {
        return res.sendStatus(401);
    }

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            return res.sendStatus(403);
        }
        req.user = user;
        next();
    });
};


const circuitOptions = {
    timeout: 5000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000,
};

const createProxy = (serviceUrl) => {
    const circuit = new CircuitBreaker(async (req) => {
        const { method, url, body, headers, id, user } = req;

        const serviceRequestHeaders = {
            'X-Request-Id': id,
        };
        if (user) {
            serviceRequestHeaders['X-User-Id'] = user.id;
        }

        try {
            const response = await axios({
                method,
                url: `${serviceUrl}${url}`,
                data: body,
                headers: serviceRequestHeaders,
                 // Forward the query parameters
                params: req.query,
            });
            return {
                status: response.status,
                data: response.data,
            };
        } catch (error) {
            if (error.response) {
                return {
                    status: error.response.status,
                    data: error.response.data,
                };
            }
            throw error;
        }
    }, circuitOptions);

    circuit.fallback(() => ({
        status: 503,
        data: { error: 'Service temporarily unavailable' },
    }));

    return (req, res) => {
        circuit.fire(req)
            .then(result => res.status(result.status).json(result.data))
            .catch(err => res.status(500).json({ error: 'Internal server error' }));
    };
};

const usersProxy = createProxy(USERS_SERVICE_URL);
const ordersProxy = createProxy(ORDERS_SERVICE_URL);

// --- Routing ---
const apiRouter = express.Router();

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'API Gateway is running' });
});

// Public routes
apiRouter.post('/users/register', usersProxy);
apiRouter.post('/users/login', usersProxy);

// Protected routes
apiRouter.use(authenticateToken);

// User routes
apiRouter.get('/users/profile', usersProxy);
apiRouter.get('/users/:id', usersProxy);

// Order routes
apiRouter.post('/orders', ordersProxy);
apiRouter.get('/orders', ordersProxy);
apiRouter.get('/orders/:orderId', ordersProxy);
apiRouter.put('/orders/:orderId', ordersProxy);


app.use('/api/v1', apiRouter);


app.listen(PORT, () => {
    console.log(`API Gateway running on port ${PORT}`);
});
