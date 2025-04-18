import pandas as pd

# Create a synthetic dataset of business data
data = {
    "id": range(1, 21),
    "business_name": [
        "TechCorp Solutions", "GreenLeaf Eco", "UrbanStyle Fashions", "DigitalDynamics",
        "HealthyHarvest Foods", "BrightFuture Education", "HomeHaven Decor",
        "EliteFitness Center", "AutoPro Mechanics", "SecureVault Systems",
        "SkyHigh Travel", "PureBeauty Spa", "QuickBite Deli", "PeakPerformance Consulting",
        "EcoFriendly Supplies", "DreamBuilders Construction", "Innovate Marketing",
        "Guardian Insurance", "SmartLiving Electronics", "FreshBloom Florists"
    ],
    "description": [
        "Innovative IT solutions provider specializing in cloud services and AI.",
        "Sustainable products promoting an eco-friendly lifestyle.",
        "Trendy and affordable clothing for urban youth.",
        "Cutting-edge digital marketing and e-commerce services.",
        "Organic and non-GMO foods for a healthier lifestyle.",
        "Education resources and e-learning platforms for all ages.",
        "Modern and stylish home decor for urban homes.",
        "State-of-the-art fitness equipment and personal training services.",
        "Comprehensive automotive repair and maintenance services.",
        "Advanced cybersecurity solutions for businesses of all sizes.",
        "Luxury travel packages and personalized trip planning.",
        "Holistic spa treatments for beauty and relaxation.",
        "Quick and delicious deli meals for busy professionals.",
        "Expert consulting services to optimize business performance.",
        "Eco-friendly office and household supplies.",
        "Custom home construction with a focus on sustainability.",
        "Creative marketing strategies for startups and small businesses.",
        "Reliable and affordable insurance plans for families and businesses.",
        "Smart home gadgets and electronics for modern living.",
        "Beautiful floral arrangements for all occasions."
    ],
    "category": [
        "Technology", "Eco Products", "Retail", "Technology",
        "Food & Beverage", "Education", "Home & Living",
        "Fitness", "Automotive", "Technology",
        "Travel", "Health & Wellness", "Food & Beverage", "Consulting",
        "Eco Products", "Construction", "Marketing", "Insurance",
        "Electronics", "Retail"
    ]
}

# Convert to a pandas DataFrame
business_data = pd.DataFrame(data)

# Save to a CSV file for user
file_path = "002_csv_semantic_search/data/business_data.csv"
business_data.to_csv(file_path, index=False)

file_path
