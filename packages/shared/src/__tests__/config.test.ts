import type * as ConfigModule from '../config.js';
import { beforeEach, expect, test, vi } from 'vitest';

const validEnv = {
  NODE_ENV: 'test',
  LOG_LEVEL: 'info',
  POSTGRES_USER: 'continuum',
  POSTGRES_PASSWORD: 'continuum',
  POSTGRES_DB: 'continuum',
  POSTGRES_PORT: '5432',
  DATABASE_URL: 'postgresql://continuum:continuum@localhost:5432/continuum?schema=public',
  SHADOW_DATABASE_URL:
    'postgresql://continuum:continuum@localhost:5432/continuum_shadow?schema=public',
  KAFKA_BROKERS: 'localhost:19092',
  KAFKA_CLIENT_ID: 'continuum',
  REDPANDA_KAFKA_PORT: '19092',
  REDPANDA_ADMIN_PORT: '9644',
  REDPANDA_CONSOLE_PORT: '8080',
  REDIS_URL: 'redis://localhost:6379',
  REDIS_PORT: '6379',
  MINIO_ROOT_USER: 'continuum',
  MINIO_ROOT_PASSWORD: 'continuum-local-dev',
  MINIO_API_PORT: '9000',
  MINIO_CONSOLE_PORT: '9001',
  S3_ENDPOINT: 'http://localhost:9000',
  S3_REGION: 'us-east-1',
  S3_ACCESS_KEY_ID: 'continuum',
  S3_SECRET_ACCESS_KEY: 'continuum-local-dev',
  S3_BUCKET_DOCUMENTS: 'continuum-documents',
  S3_BUCKET_MODELS: 'continuum-models',
  S3_FORCE_PATH_STYLE: 'true',
  EMBEDDING_MODEL: 'continuum/hash-embedding-demo',
  EMBEDDING_DIM: '384',
  DRIFT_THRESHOLD: '0.35',
  OTEL_SERVICE_NAME: 'continuum',
  OTEL_EXPORTER_OTLP_ENDPOINT: '',
};

beforeEach(() => {
  vi.resetModules();
  for (const [key, value] of Object.entries(validEnv)) {
    process.env[key] = value;
  }
});

test('validates and coerces runtime configuration', async () => {
  const { validateConfig }: typeof ConfigModule = await import('../config.js');

  const config = validateConfig(validEnv);

  expect(config.NODE_ENV).toBe('test');
  expect(config.POSTGRES_PORT).toBe(5432);
  expect(config.S3_FORCE_PATH_STYLE).toBe(true);
  expect(config.DRIFT_THRESHOLD).toBe(0.35);
});

test('exits immediately for invalid configuration', async () => {
  const { validateConfig }: typeof ConfigModule = await import('../config.js');
  const exit = vi.spyOn(process, 'exit').mockImplementation((): never => {
    throw new Error('process.exit');
  });
  const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);

  expect(() => validateConfig({ ...validEnv, DATABASE_URL: 'not-a-url' })).toThrow('process.exit');
  expect(exit).toHaveBeenCalledWith(1);
  expect(error).toHaveBeenCalled();
});
