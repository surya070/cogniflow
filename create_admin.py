"""Run once to create the admin account: python create_admin.py"""
from app import app, db
from models import User

ADMIN_EMAIL    = "admin@cogniflow.com"
ADMIN_PASSWORD = "Cogniflow@2026"
ADMIN_NAME     = "Admin"

with app.app_context():
    existing = User.query.filter_by(email=ADMIN_EMAIL).first()
    if existing:
        print(f"Admin already exists (id={existing.id}). Updating password.")
        existing.set_password(ADMIN_PASSWORD)
        existing.role = "admin"
        db.session.commit()
        print("Done.")
    else:
        user = User(
            email=ADMIN_EMAIL,
            name=ADMIN_NAME,
            employee_id=ADMIN_EMAIL,
            role="admin",
            webhook_token=User.make_webhook_token(),
        )
        user.set_password(ADMIN_PASSWORD)
        db.session.add(user)
        db.session.commit()
        print(f"Admin created (id={user.id}).")
        print(f"  Email   : {ADMIN_EMAIL}")
        print(f"  Password: {ADMIN_PASSWORD}")
