# FDM SmartLens API Documentation

## Base URL

```
Production: https://api.fdmsmartlens.com/v1
Development: https://dev-api.fdmsmartlens.com/v1
```

## Authentication

API requests require authentication using Bearer token:

```
Authorization: Bearer <your_token>
```

## Endpoints

### Diagnoses

#### Upload Diagnosis Result

```http
POST /diagnoses
Content-Type: application/json
Authorization: Bearer <token>

{
  "deviceId": "device_123",
  "diagnosis": {
    "diseaseName": "바이러스성출혈성패혈증",
    "diseaseCode": 1,
    "confidence": 0.87,
    "symptoms": [
      {
        "name": "Bleeding",
        "confidence": 0.92,
        "boundingBox": {
          "x1": 0.1,
          "y1": 0.2,
          "x2": 0.5,
          "y2": 0.8
        }
      }
    ]
  },
  "imageUrl": "https://storage.fdmsmartlens.com/images/xxx.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "diag_1234567890"
  }
}
```

#### Get Diagnosis History

```http
GET /diagnoses?page=1&pageSize=20
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "diagnoses": [...],
    "total": 156,
    "page": 1,
    "pageSize": 20
  }
}
```

#### Get Single Diagnosis

```http
GET /diagnoses/{id}
Authorization: Bearer <token>
```

#### Delete Diagnosis

```http
DELETE /diagnoses/{id}
Authorization: Bearer <token>
```

### Statistics

#### Get Statistics

```http
GET /statistics
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalDiagnoses": 156,
    "diseaseDistribution": {
      "정상": 114,
      "바이러스성출혈성패혈증": 12,
      "림포시스티스병": 8
    },
    "averageConfidence": 0.87,
    "recentDiagnoses": [...]
  }
}
```

### Images

#### Upload Image

```http
POST /images
Content-Type: multipart/form-data
Authorization: Bearer <token>

FormData:
  image: <image_file>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "imageUrl": "https://storage.fdmsmartlens.com/images/xxx.jpg"
  }
}
```

### User

#### Get User Info

```http
GET /user
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "Username",
    "createdAt": 1704067200000
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "success": false,
  "error": "Invalid request parameters"
}
```

### 401 Unauthorized
```json
{
  "success": false,
  "error": "Invalid or expired token"
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal server error"
}
```

## Rate Limiting

- Free tier: 100 requests per minute
- Premium tier: 1000 requests per minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067260
```

## SDK Usage

### React Native

```typescript
import ApiService from './services/ApiService';

const api = ApiService.getInstance();
api.setDeviceId('device_123');
api.setAuthToken('your_token');

// Upload diagnosis
const result = await api.uploadDiagnosis({
  deviceId: 'device_123',
  diagnosis: { ... }
});

// Get history
const history = await api.getDiagnosisHistory(1, 20);

// Get statistics
const stats = await api.getStatistics();
```

### Android (Kotlin)

```kotlin
// Using Retrofit or similar HTTP client
val apiClient = ApiClient.getInstance()
apiClient.setAuthToken("your_token")

// Upload diagnosis
val result = apiClient.uploadDiagnosis(diagnosisRequest)

// Get history
val history = apiClient.getDiagnosisHistory(page = 1, pageSize = 20)
```
