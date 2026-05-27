web: python manage.py migrate --noinput && (python manage.py createsuperuser --noinput || true) && exec gunicorn config.wsgi:application --timeout 120
release: python manage.py migrate --noinput
