web: gunicorn config.wsgi:application
release: python manage.py migrate --noinput && python manage.py createsuperuser --noinput || true
