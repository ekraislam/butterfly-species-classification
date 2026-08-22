# -*- coding: utf-8 -*-
"""
Bilingual UI Translations & Rich 360-Degree Species Dossiers
For AI Butterfly Vision Lab (ResNet-18 + Grad-CAM XAI)
Languages Supported: English (EN) & Bengali (BN)
"""

UI_TEXT = {
    "EN": {
        # App Header & Nav
        "app_title": "AI Butterfly Vision",
        "app_subtitle": "Ultra-High Precision Butterfly Species Identification with Real-Time Explainable AI (XAI)",
        "hero_badge": "🦋 ResNet-18 • Deep Transfer Learning • Grad-CAM XAI",
        "hero_tag_species": "🌿 8 Tropical Species",
        "hero_tag_xai": "🔬 Real-Time Grad-CAM XAI",
        "hero_tag_acc": "⚡ 97.22% Precision",
        "hero_tag_cert": "📄 4K Report Generator",
        "btn_install_app": "📲 Install App",
        "install_modal_title": "📲 Install AI Butterfly Vision on Your Device",
        "install_modal_sub": "Install this web app on your phone or desktop to access it instantly like a native app!",
        "install_android_title": "🤖 Android & Chrome / Desktop",
        "install_android_step1": "1. Tap the <strong>⋮ (3-dots menu)</strong> at the top right (or click <strong>Install</strong> on the address bar).",
        "install_android_step2": "2. Select <strong>'Install app'</strong> or <strong>'Add to Home screen'</strong>.",
        "install_android_step3": "3. Tap <strong>'Install'</strong> to enjoy the full-screen native experience!",
        "install_ios_title": "🍎 iPhone & iPad (Safari)",
        "install_ios_step1": "1. Tap the <strong>Share button ( ⬆️ )</strong> at the bottom bar in Safari.",
        "install_ios_step2": "2. Scroll down and tap <strong>'Add to Home Screen'</strong>.",
        "install_ios_step3": "3. Tap <strong>'Add'</strong> at the top right to create the home screen icon.",
        
        # Metric Strip
        "metric_acc_val": "97.22%",
        "metric_acc_lbl": "Model Precision",
        "metric_f1_val": "0.9742",
        "metric_f1_lbl": "Macro F1-Score",
        "metric_species_val": "8 Species",
        "metric_species_lbl": "Taxa Coverage",
        "metric_xai_val": "Grad-CAM",
        "metric_xai_lbl": "Explainable AI",
        
        # Input Studio
        "studio_title": "📸 Select or Capture Specimen Image",
        "tab_gallery": "⚡ 1-Click Benchmark Gallery",
        "tab_upload": "📤 Upload Specimen Photo",
        "tab_camera": "📷 Live Camera Capture",
        "gallery_prompt": "Click any benchmark test specimen below to instantly load and evaluate:",
        "upload_title": "📥 Drag and drop your butterfly image here",
        "upload_sub": "Limit 200MB per file • JPG, JPEG, PNG, WEBP, AVIF, HEIC supported",
        "upload_help": "Upload a clear photo of a butterfly for neural analysis",
        "camera_prompt": "Point your webcam or mobile camera directly at the butterfly specimen:",
        "camera_label": "Take a photo of the butterfly",
        "cam_ready_title": "Live Field Lens Ready",
        "cam_ready_sub": "Activate your device camera to capture live butterfly specimens directly in the field",
        "btn_activate_cam": "📸 Activate Live Camera",
        "btn_deactivate_cam": "🛑 Turn Off Camera",
        
        # Specimen Command Bar
        "active_specimen": "📸 Active Specimen",
        "ready_badge": "✨ Ready for Neural Analysis",
        "deep_feat_title": "Deep Feature Extraction & Explainable AI",
        "deep_feat_desc": "ResNet-18 extracts fine-grained wing textures, venation patterns, and optical scales, followed by Grad-CAM backward gradient computation.",
        "btn_run_analysis": "✨ Run Neural Analysis",
        "btn_reset": "🔄 Reset / Clear",
        "btn_reset_help": "Clear the active specimen and reset analysis",
        "spinner_msg": "Computing Neural Feature Maps & Gradient Backpropagation...",
        "msg_choose_specimen": "💡 Choose a butterfly specimen from the 1-Click Gallery, Upload, or Live Camera above to begin analysis.",
        "msg_run_prompt": "👆 Click **✨ Run Neural Analysis** above to classify this specimen and generate Grad-CAM heatmaps.",
        
        # Results Section
        "result_heading": "🎯 Classification Diagnostics",
        "identified_species": "Identified Species",
        "scientific_name_lbl": "Scientific Name",
        "family_lbl": "Taxonomic Family",
        "confidence_gauge_lbl": "Confidence Score",
        "top3_title": "📊 Prediction Probability Hierarchy",
        
        # XAI Section
        "xai_heading": "🔬 Explainable AI (Grad-CAM) Visualizer",
        "slider_label": "🎨 Heatmap Intensity Blend:",
        "slider_help": "Adjust slider to blend Grad-CAM heatmap with original specimen",
        "ch_original": "📸 Original",
        "ch_cam": "🔥 Heatmap",
        "ch_overlay": "✨ AI Overlay",
        "ch_compare": "🔍 Compare",
        "xai_box_title": "💡 Neural Attention Explanation",
        "btn_inspect_report": "📄 Generate & Inspect AI Report Certificate",
        
        # Certificate View
        "cert_inspection_tag": "Official Specimen Inspection Certificate",
        "cert_preview_title": "AI Inspection Certificate Preview",
        "cert_caption": "Ultra-HD Verification Certificate • Specimen: {species} • Verified by ResNet-18 & Grad-CAM",
        "btn_download_cert": "📥 Download Official Certificate (High-Res PNG)",
        "btn_close_cert": "❌ Close Certificate Preview",
        
        # Species Roster
        "roster_heading": "🌿 Supported Species & Bio-Taxonomy Directory",
        "roster_sub": "Our neural network is trained on 8 distinct butterfly species. Click **Explore Dossier** on any card to read its full profile and test with 1-click:",
        "btn_explore_dossier": "📖 Explore Dossier",
        "btn_test_species": "🚀 Test This Species in AI Vision Lab",
        
        # Dossier Modal Headings
        "dossier_modal_tag": "Taxonomic Specimen Bio-Dossier",
        "dossier_habitat": "🌍 Natural Habitat & Distribution",
        "dossier_appearance": "🎨 Visual Morphology & Wing Patterns",
        "dossier_superpower": "🧬 Biological Marvel & Defense Strategies",
        "dossier_xai": "🔬 AI Neural Attention Highlights",
        "dossier_stat_wingspan": "Wingspan",
        "dossier_stat_lifespan": "Avg. Lifespan",
        "dossier_stat_family": "Family",
        "dossier_stat_precision": "Model Precision",
        
        # Footer
        "footer_designed_by": "Designed & Developed by",
        "footer_tech": "Deep Transfer Learning (ResNet-18) • Native PyTorch Grad-CAM • TorchScript Mobile & Edge Architecture"
    },
    
    "BN": {
        # App Header & Nav
        "app_title": "এআই বাটারফ্লাই ভিশন",
        "app_subtitle": "কৃত্রিম বুদ্ধিমত্তা (AI) ও যুক্তিনির্ভর ব্যাখ্যা (XAI) সহ অত্যন্ত নির্ভুল প্রজাপতি শনাক্তকরণ ব্যবস্থা",
        "hero_badge": "🦋 ডিপ লার্নিং • রেসনেট-১৮ • গ্র্যাড-ক্যাম XAI",
        "hero_tag_species": "🌿 ৮টি বিশেষ ক্রান্তীয় প্রজাতি",
        "hero_tag_xai": "🔬 লাইভ এআই দৃষ্টি (Grad-CAM)",
        "hero_tag_acc": "⚡ ৯৭.২২% শনাক্তকরণ নির্ভুলতা",
        "hero_tag_cert": "📄 অফিশিয়াল রিপোর্ট সার্টিফিকেট",
        "btn_install_app": "📲 অ্যাপ ইনস্টল করুন",
        "install_modal_title": "📲 আপনার ডিভাইসে অ্যাপ হিসেবে ইনস্টল করুন",
        "install_modal_sub": "ব্রাউজারের কোনো বাড়তি ঝামেলা ছাড়াই যেকোনো সময় সরাসরি ফুল-স্ক্রিন অ্যাপ হিসেবে চালান!",
        "install_android_title": "🤖 অ্যান্ড্রয়েড (Chrome) ও ডেস্কটপ",
        "install_android_step1": "১. ব্রাউজারের উপরে ডানে <strong>⋮ (৩-ডট মেনু)</strong>-তে চাপ দিন (অথবা অ্যাড্রেস বারের <strong>Install</strong> আইকনে চাপ দিন)।",
        "install_android_step2": "২. মেনু থেকে <strong>'Install app'</strong> অথবা <strong>'Add to Home screen'</strong> বেছে নিন।",
        "install_android_step3": "৩. <strong>'Install'</strong> বাটনে চাপ দিলেই হোমস্ক্রিনে অ্যাপটি চলে আসবে!",
        "install_ios_title": "🍎 আইফোন ও আইপ্যাড (Safari)",
        "install_ios_step1": "১. Safari ব্রাউজারের নিচে থাকা <strong>Share বোতাম ( ⬆️ )</strong>-এ চাপ দিন।",
        "install_ios_step2": "২. মেনু থেকে নিচের দিকে স্ক্রোল করে <strong>'Add to Home Screen'</strong> বেছে নিন।",
        "install_ios_step3": "৩. উপরে ডানে <strong>'Add'</strong> বাটনে চাপ দিন।",
        
        # Metric Strip
        "metric_acc_val": "৯৭.২২%",
        "metric_acc_lbl": "মডেলের নির্ভুলতা (Precision)",
        "metric_f1_val": "০.৯৭৪২",
        "metric_f1_lbl": "গুণগত F1-স্কোর",
        "metric_species_val": "৮টি প্রজাতি",
        "metric_species_lbl": "শনাক্তকরণ ক্ষমতা",
        "metric_xai_val": "Grad-CAM",
        "metric_xai_lbl": "এআই যুক্তি ও প্রমাণ",
        
        # Input Studio
        "studio_title": "📸 প্রজাপতির ছবি নির্বাচন বা সরাসরি ক্যাপচার করুন",
        "tab_gallery": "⚡ ১-ক্লিকে নমুনা গ্যালারি",
        "tab_upload": "📤 নিজের ছবি আপলোড করুন",
        "tab_camera": "📷 সরাসরি ক্যামেরা দিয়ে ছবি তুলুন",
        "gallery_prompt": "নিচের যেকোনো প্রজাপতির কার্ডে ক্লিক করে সরাসরি পরীক্ষা করুন:",
        "upload_title": "📥 প্রজাপতির ছবি এখানে এনে ছেড়ে দিন (Drag & Drop)",
        "upload_sub": "অথবা বক্সের ভেতর <strong>Browse files</strong> বাটনে ক্লিক করে ডিভাইস থেকে ছবি বেছে নিন • JPG, PNG, WEBP, AVIF সমর্থিত",
        "upload_help": "যেকোনো স্পষ্ট প্রজাপতির ছবি সরাসরি এই বক্সে ছেড়ে দিন অথবা মেমোরি থেকে বেছে নিন",
        "camera_prompt": "আপনার মোবাইল বা ল্যাপটপের ক্যামেরা প্রজাপতির দিকে তাক করে ছবি তুলুন:",
        "camera_label": "📸 ক্যামেরার বোতাম চেপে ছবি তুলুন",
        "cam_ready_title": "লাইভ ক্যামেরা লেন্স প্রস্তুত",
        "cam_ready_sub": "আপনার ডিভাইসের ক্যামেরা দিয়ে সরাসরি জীবন্ত প্রজাপতির ছবি তুলতে নিচের বাটনে চাপ দিন",
        "btn_activate_cam": "📸 ক্যামেরা চালু করুন",
        "btn_deactivate_cam": "🛑 ক্যামেরা বন্ধ করুন",
        
        # Specimen Command Bar
        "active_specimen": "📸 নির্বাচিত প্রজাপতি",
        "ready_badge": "✨ এআই বিশ্লেষণের জন্য প্রস্তুত",
        "deep_feat_title": "ডিপ নিউরাল বিশ্লেষণ ও এআই দৃষ্টি",
        "deep_feat_desc": "উন্নত রেসনেট-১৮ মডেল প্রজাপতির ডানার সূক্ষ্ম নকশা, রঙ এবং শিরার বিন্যাস খতিয়ে দেখে প্রজাতি শনাক্ত করবে এবং এআই কীভাবে সিদ্ধান্ত নিলো তা হিটম্যাপে দেখাবে।",
        "btn_run_analysis": "✨ এআই দিয়ে প্রজাতি শনাক্ত করুন",
        "btn_reset": "🔄 ছবি রিসেট / মুছুন",
        "btn_reset_help": "বর্তমান ছবি সরিয়ে নতুন ছবি বেছে নিন",
        "spinner_msg": "এআই মডেল প্রজাপতির ডানার গঠন ও বিশেষ বৈশিষ্ট্য বিশ্লেষণ করছে...",
        "msg_choose_specimen": "💡 শুরু করতে উপরে ১-ক্লিক গ্যালারি, নিজের ছবি আপলোড বা লাইভ ক্যামেরা থেকে একটি প্রজাপতির ছবি নির্বাচন করুন।",
        "msg_run_prompt": "👆 প্রজাতি শনাক্ত ও এআই ব্যাখ্যা দেখতে উপরে **✨ এআই দিয়ে প্রজাতি শনাক্ত করুন** বাটনে ক্লিক করুন।",
        
        # Results Section
        "result_heading": "🎯 শনাক্তকরণের চূড়ান্ত ফলাফল",
        "identified_species": "শনাক্তকৃত প্রজাতি",
        "scientific_name_lbl": "বৈজ্ঞানিক নাম",
        "family_lbl": "গোত্র / পরিবার",
        "confidence_gauge_lbl": "এআই নিশ্চয়তার হার (Confidence)",
        "top3_title": "📊 শীর্ষ ৩টি সম্ভাব্য প্রজাতির তালিকা",
        
        # XAI Section
        "xai_heading": "🔬 এআই কীভাবে শনাক্ত করলো? (Grad-CAM ভিজ্যুয়ালাইজার)",
        "slider_label": "🎨 হিটম্যাপের তীব্রতা নিয়ন্ত্রণ স্লাইডার:",
        "slider_help": "স্লাইডার টেনে এআই হিটম্যাপ ও আসল ছবির মিশ্রণ নিয়ন্ত্রণ করুন",
        "ch_original": "📸 মূল ছবি",
        "ch_cam": "🔥 এআই হিটম্যাপ",
        "ch_overlay": "✨ বিশ্লেষণ ওভারলে",
        "ch_compare": "🔍 পাশাপাশি তুলনা",
        "xai_box_title": "💡 এআই সিদ্ধান্তের পেছনের মূল যুক্তি ও নজর",
        "btn_inspect_report": "📄 অফিশিয়াল এআই রিপোর্ট সার্টিফিকেট তৈরি করুন",
        
        # Certificate View
        "cert_inspection_tag": "অফিশিয়াল নমুনা পরিদর্শন ও এআই সার্টিফিকেট",
        "cert_preview_title": "এআই পরিদর্শন সার্টিফিকেটের প্রিভিউ",
        "cert_caption": "আল্ট্রা-এইচডি অফিসিয়াল সার্টিফিকেট • নমুনা: {species} • রেসনেট-১৮ ও Grad-CAM দ্বারা পরীক্ষিত",
        "btn_download_cert": "📥 সার্টিফিকেট ডাউনলোড করুন (হাই-রেজোলিউশন PNG)",
        "btn_close_cert": "❌ প্রিভিউ বন্ধ করুন",
        
        # Species Roster
        "roster_heading": "🌿 আমাদের শনাক্তকরণ ক্ষমতার ৮টি বিশেষ প্রজাপতি",
        "roster_sub": "আমাদের এআই মডেলটি নিচের ৮টি বিশেষ ক্রান্তীয় প্রজাপতিকে চিনতে দক্ষ। বিস্তারিত জীবনবৃত্তান্ত ও বৈশিষ্ট্য জানতে যেকোনো কার্ডের **সম্পূর্ণ প্রোফাইল** বাটনে ক্লিক করুন:",
        "btn_explore_dossier": "📖 সম্পূর্ণ প্রোফাইল ও তথ্য",
        "btn_test_species": "🚀 এই প্রজাপতিটি এআই দিয়ে টেস্ট করুন",
        
        # Dossier Modal Headings
        "dossier_modal_tag": "প্রজাপতির পূর্ণাঙ্গ পরিচিতি ও এআই বিশ্লেষণ",
        "dossier_habitat": "🌍 প্রাকৃতিক আবাসস্থল ও ভৌগোলিক বিচরণ",
        "dossier_appearance": "🎨 ডানার রঙ, গঠন ও বিশেষ চিহ্ন",
        "dossier_superpower": "🧬 প্রকৃতিতে বেঁচে থাকার কৌশল ও আত্মরক্ষা",
        "dossier_xai": "🔬 এআই মডেল যেভাবে এই প্রজাপতিকে চেনে",
        "dossier_stat_wingspan": "ডানার বিস্তার",
        "dossier_stat_lifespan": "গড় আয়ু",
        "dossier_stat_family": "গোত্র / পরিবার",
        "dossier_stat_precision": "শনাক্তকরণ নির্ভুলতা",
        
        # Footer
        "footer_designed_by": "পরিকল্পনা ও বাস্তবায়নে",
        "footer_brand": "এআই বাটারফ্লাই ভিশন ল্যাব"
    }
}

# -----------------------------------------------------------------------------
# Rich 360-Degree Bilingual Species Dossiers (8 Tropical Species)
# -----------------------------------------------------------------------------
SPECIES_DOSSIER = {
    "ADONIS": {
        "name_en": "Adonis Blue",
        "name_bn": "অ্যাডনিস ব্লু (নীলপরী)",
        "scientific_name": "Lysandra bellargus",
        "family_en": "Lycaenidae (Gossamer-winged butterflies)",
        "family_bn": "লাইকেনিডি (পাতলা রেশমি ডানা)",
        "color_primary": "#0284C7",
        "wingspan": "30 – 36 mm",
        "lifespan_en": "2 – 3 weeks (adult stage)",
        "lifespan_bn": "২ – ৩ সপ্তাহ (পূর্ণাঙ্গ অবস্থা)",
        "habitat_en": "Sheltered, sunny south-facing chalk and limestone grasslands across Southern & Central Europe and temperate Asia.",
        "habitat_bn": "দক্ষিণ ও মধ্য ইউরোপ এবং এশিয়ার রৌদ্রোজ্জ্বল চুনাপাথরযুক্ত পাহাড়ী তৃণভূমি।",
        "appearance_en": "Males display intensely vibrant sky-blue dorsal wings bordered by a distinctive checkered black-and-white fringe. Females are chocolate-brown with subtle orange submarginal crescents.",
        "appearance_bn": "পুরুষ প্রজাপতির ডানা অসম্ভব উজ্জ্বল আসমানী নীল এবং ডানার প্রান্তে সাদা-কালো ছকের মতো সূক্ষ্ম বর্ডার থাকে। স্ত্রী প্রজাপতি উষ্ণ গাঢ় বাদামী রঙের হয়।",
        "superpower_en": "Symbiotic Ant Alliance: Caterpillars produce sweet amino-acid secretions from special glands to attract red ants (Myrmica sabuleti), which fiercely guard the larvae from predators in exchange for sugar nectar.",
        "superpower_bn": "পিঁপড়ার সাথে বিশেষ বন্ধুত্ব: এদের শুঁয়োপোকা শরীর থেকে মিষ্টি তরল নিঃসরণ করে এক বিশেষ প্রজাতির লাল পিঁপড়াকে আকৃষ্ট করে। পিঁপড়ারা মধুর লোভে শুঁয়োপোকাকে শিকারীদের হাত থেকে পাহারা দেয়।",
        "ai_attention_en": "The ResNet-18 neural model heavily focuses on the checkered white wing fringes, micro-scale venous reflections, and vibrant sky-blue dorsal iridescence.",
        "ai_attention_bn": "আমাদের এআই মডেল ডানার প্রান্তের সাদা-কালো ছক কাটা বর্ডার এবং উজ্জ্বল আসমানী নীল ডানার প্রতিফলনের ওপর সর্বোচ্চ ফোকাস করে শনাক্ত করে।",
        "model_precision": "88.44%"
    },
    
    "CLODIUS PARNASSIAN": {
        "name_en": "Clodius Parnassian",
        "name_bn": "ক্লোডিয়াস পার্নাসিয়ান (তুষার)",
        "scientific_name": "Parnassius clodius",
        "family_en": "Papilionidae (Alpine Swallowtails)",
        "family_bn": "প্যাপিলিওনিডি (সোয়ালোটেল)",
        "color_primary": "#E11D48",
        "wingspan": "55 – 70 mm",
        "lifespan_en": "3 – 4 weeks",
        "lifespan_bn": "৩ – ৪ সপ্তাহ",
        "habitat_en": "High alpine meadows, mountain clearings, and subalpine coniferous forest borders of Western North America (Cascades and Sierra Nevada).",
        "habitat_bn": "পশ্চিম উত্তর আমেরিকার ক্যাসকেডস ও সিয়েরা নেভাদা পর্বতমালার উঁচু বরফাবৃত তৃণভূমি ও পাইন বনানী।",
        "appearance_en": "Semi-translucent chalk-white parchment wings adorned with striking crimson-red eye-spots (ocelli) encircled by bold black rings on hindwings, and charcoal-dusted wing margins.",
        "appearance_bn": "কাগজের মতো আধা-স্বচ্ছ সাদা রেশমি ডানা, যার পেছনের অংশে কালো বৃত্ত দিয়ে ঘেরা স্পষ্ট টকটকে লাল চোখের মতো স্পট (Ocelli) থাকে।",
        "superpower_en": "Sub-Zero Alpine Resilience: Its dense, fur-like thoracic setae trap body heat in freezing alpine altitudes, while dark wing veins rapidly absorb low-angle high-altitude solar radiation.",
        "superpower_bn": "হিমশীতল সহ্যক্ষমতা: এদের শরীরের ঘন লোম বরফশীতল পাহাড়ি বাতাসে শরীরের তাপমাত্রা ধরে রাখে এবং কালো শিরাগুলো সূর্যালোক দ্রুত শোষণ করে শরীর গরম রাখে।",
        "ai_attention_en": "The model zeroes in on the high-contrast crimson ocelli, semi-translucent wing edges, and the charcoal venation density in the forewing cells.",
        "ai_attention_bn": "এআই মডেল পেছনের ডানার উজ্জ্বল লাল বৃত্তাকার স্পট এবং আধা-স্বচ্ছ ডানার প্রান্তের শিরার প্যাটার্ন দেখে নির্ভুলভাবে এটিকে শনাক্ত করে।",
        "model_precision": "92.15%"
    },
    
    "GREEN CELLED CATTLEHEART": {
        "name_en": "Green Celled Cattleheart",
        "name_bn": "গ্রিন সেলড ক্যাটলহার্ট (পান্না হৃদয়)",
        "scientific_name": "Parides sesostris",
        "family_en": "Papilionidae (Swallowtails)",
        "family_bn": "প্যাপিলিওনিডি (সোয়ালোটেল)",
        "color_primary": "#059669",
        "wingspan": "90 – 110 mm",
        "lifespan_en": "4 – 6 weeks",
        "lifespan_bn": "৪ – ৬ সপ্তাহ",
        "habitat_en": "Dense tropical rainforest canopies and riverine understories spanning from southern Mexico through Central America to the Amazon Basin.",
        "habitat_bn": "মেক্সিকো থেকে শুরু করে মধ্য আমেরিকা এবং আমাজন রেইনফরেস্টের গহিন চিরসবুজ বনাঞ্চল।",
        "appearance_en": "Velvety pitch-black forewings illuminated by a glowing emerald-green central patch; hindwings showcase a crescent constellation of vibrant ruby-red spots.",
        "appearance_bn": "ভেলভেটের মতো গভীর কুচকুচে কালো ডানা, যার মাঝখানে জ্বলজ্বলে পান্না-সবুজ রঙের প্যাচ এবং নিচের ডানায় রুবি-লাল রত্নের মতো সুন্দর স্পট থাকে।",
        "superpower_en": "Aristolochic Toxic Arsenal: The caterpillars feed exclusively on poisonous Dutchman's Pipe vines (Aristolochia), sequestering toxic aristolochic acids that make both larvae and adults completely unpalatable and deadly to birds.",
        "superpower_bn": "বিষাক্ত আত্মরক্ষা: এদের শুঁয়োপোকা বিষাক্ত ডাচম্যানস পাইপ লতা খেয়ে শরীরে বিষ সঞ্চয় করে রাখে, যার ফলে কোনো পাখি এদের খেলে মারাত্মক বিষক্রিয়ায় আক্রান্ত হয় এবং এদের ছোঁয় না।",
        "ai_attention_en": "Neural gradients strongly highlight the luminous emerald-green forewing patch and its sharp contrast against the matte-black wing background.",
        "ai_attention_bn": "এআই মডেল ভেলভেট কালো ডানার ওপর অবস্থিত উজ্জ্বল পান্না-সবুজ প্যাচ এবং পেছনের লাল ডটের তীব্র কনট্রাস্ট চিহ্নিত করে।",
        "model_precision": "94.60%"
    },
    
    "MONARCH": {
        "name_en": "Monarch",
        "name_bn": "মোনার্ক বাটারফ্লাই (রাজকীয়)",
        "scientific_name": "Danaus plexippus",
        "family_en": "Nymphalidae (Brush-footed butterflies)",
        "family_bn": "নিমফ্যালিডি (ব্রাশ-ফুটেড)",
        "color_primary": "#EA580C",
        "wingspan": "89 – 102 mm",
        "lifespan_en": "Up to 8 months (Migratory generation)",
        "lifespan_bn": "৮ মাস পর্যন্ত (মাইগ্রেটরি বংশধর)",
        "habitat_en": "Open fields, meadows, roadsides, and coastal reserves across the Americas; famous for overwintering in the high-altitude Oyamel fir forests of Michoacán, Mexico.",
        "habitat_bn": "উত্তর ও দক্ষিণ আমেরিকার বিস্তৃত প্রান্তর এবং মেক্সিকোর মিকোয়াকানের উচ্চ पर्वतমালা যেখানে এরা শীতকালে আশ্রয় নেয়।",
        "appearance_en": "Tawny fiery orange upperside etched with bold black structural veins, surrounded by a wide black border studded with a double constellation of bright white dots.",
        "appearance_bn": "উজ্জ্বল আগুন-কমলা রঙের ডানা, যাতে গাঢ় কালো রেখায় আঁকা শিরা এবং ডানার কিনারায় দুই সারি উজ্জ্বল সাদা ফোঁটার নকশা থাকে।",
        "superpower_en": "Epic 3,000-Mile Migration: The autumn 'Super Generation' flies up to 3,000 miles from Canada to Mexican mountain peaks, navigating using a genetically programmed time-compensated solar compass and magnetic sensors.",
        "superpower_bn": "৩,০০০ মাইলের অলৌকিক যাত্রা: প্রতি শীতে এদের এক বিশেষ প্রজন্ম কানাডা থেকে একটানা উড়ে ৩,০০০ মাইল পাড়ি দিয়ে মেক্সিকোর পাহাড়ে পৌঁছায়। এরা সূর্যের আলো ও পৃথিবীর চৌম্বকক্ষেত্র ব্যবহার করে দিক চেনে!",
        "ai_attention_en": "Model activations heavily prioritize the intricate geometric venation network, the thick black wing margins, and the double row of submarginal white dots.",
        "ai_attention_bn": "এআই মডেল ডানার কালো শিরার জটিল জ্যামিতিক বিন্যাস এবং ডানার প্রান্তের জোড়া সাদা বিন্দুর প্যাটার্নকে সর্বোচ্চ প্রাধান্য দেয়।",
        "model_precision": "97.55%"
    },
    
    "ORANGE OAKLEAF": {
        "name_en": "Orange Oakleaf",
        "name_bn": "অরেঞ্জ ওকলিফ (শুকনো পাতা)",
        "scientific_name": "Kallima inachus",
        "family_en": "Nymphalidae (Brush-footed butterflies)",
        "family_bn": "নিমফ্যালিডি (পাতা-অনুকরণকারী)",
        "color_primary": "#D97706",
        "wingspan": "85 – 110 mm",
        "lifespan_en": "3 – 5 weeks",
        "lifespan_bn": "৩ – ৫ সপ্তাহ",
        "habitat_en": "Dense tropical and subtropical moist broadleaf forests along riverbanks across East Asia, the Himalayas, and Southeast Asia.",
        "habitat_bn": "হিমালয় পর্বতমালা, ভারত, চীন ও দক্ষিণ-পূর্ব এশিয়ার ঘন ক্রান্তীয় আর্দ্র বনাঞ্চল ও নদীর তীর।",
        "appearance_en": "Incredible dual-nature morphology: Upperside reveals rich metallic cobalt-blue with a glowing fiery orange diagonal sash. Underside is an astonishingly perfect replica of a decaying dead oak leaf complete with midrib, veins, fungal decay spots, and leaf stem.",
        "appearance_bn": "প্রকৃতির চরম দ্বৈত রূপ: ডানা খুললে দেখা যায় উজ্জ্বল নীল ও জ্বলন্ত কমলার অসাধারণ সৌন্দর্য; কিন্তু ডানা বন্ধ করলেই তা ১০০% শুকনো মরা পাতার রূপ ধারণ করে, যার মধ্যে পাতার শিরা, বোঁটা ও ছত্রাকের দাগ পর্যন্ত স্পষ্ট!",
        "superpower_en": "Ultimate Crypsis Mastery: When perched with closed wings on a tree branch, it becomes 100% invisible to avian predators, effortlessly mimicking decayed plant matter with zero visual giveaway.",
        "superpower_bn": "অদৃশ্য হওয়ার অলৌকিক ক্ষমতা: গাছের ডালে ডানা বন্ধ করে বসলে মানুষ বা শিকারী পাখি কারো পক্ষেই একে মরা শুকনো পাতা ছাড়া অন্য কিছু ভাবা অসম্ভব!",
        "ai_attention_en": "Neural attention locks onto the angled apex of the forewing, the midrib line illusion, and the high-contrast boundary between the blue-orange dorsal sash.",
        "ai_attention_bn": "এআই মডেল ডানার খাঁজকাটা তীক্ষ্ণ কোণ, পাতার শিরার মতো দাগ এবং কমলার সাথে নীল রঙের সীমানা নিখুঁতভাবে শনাক্ত করে।",
        "model_precision": "95.10%"
    },
    
    "PAPER KITE": {
        "name_en": "Paper Kite",
        "name_bn": "পেপার কাইট (কাগজের ঘুড়ি)",
        "scientific_name": "Idea leuconoe",
        "family_en": "Nymphalidae (Danainae • Tree Nymphs)",
        "family_bn": "নিমফ্যালিডি (গাছের পরি)",
        "color_primary": "#0284C7",
        "wingspan": "120 – 140 mm",
        "lifespan_en": "1 – 3 months",
        "lifespan_bn": "১ – ৩ মাস",
        "habitat_en": "Coastal mangrove forests, lowland rainforest clearings, and tropical butterfly sanctuaries across Southeast Asia, Taiwan, and the Philippines.",
        "habitat_bn": "ফিলিপাইন, তাইওয়ান এবং দক্ষিণ-পূর্ব এশিয়ার উপকূলীয় ম্যানগ্রোভ বন ও ক্রান্তীয় রেইনফরেস্ট।",
        "appearance_en": "Enormous semi-translucent parchment-white wings overlaid with bold, intricate black veins and large artistic ink-blot margin spots.",
        "appearance_bn": "বিশাল আকৃতির আধা-স্বচ্ছ সাদা রেশমি ডানা, যাতে কালো রঙের নাটকীয় শিরার জাল এবং কালির ফোঁটার মতো বড় বড় কালো স্পট থাকে।",
        "superpower_en": "Paper-Glider Aerodynamics & Golden Chrysalis: Boasts one of the lowest wing-loading ratios among all Lepidoptera, allowing it to float effortlessly in the air like a piece of rice paper. Its pupa is pure reflective mirror-gold!",
        "superpower_bn": "কাগজের মতো ভাসা ও খাঁটি সোনার পিউপা: বাতাসের চেয়েও হালকা ভঙ্গিতে কাগজের ঘুড়ির মতো ভেসে বেড়ায়। সবচেয়ে অবাক করা বিষয় হলো, এদের গুটি বা পিউপা দেখতে হুবহু খাঁটি চকচকে সোনার মতো!",
        "ai_attention_en": "The model detects the high-contrast black grid venation, translucent white wing cells, and large ink-like submarginal spots across the expansive wing surface.",
        "ai_attention_bn": "এআই মডেল সাদা ডানার ভেতর কালো জালিকাকার শিরা এবং ডানার কিনারার বড় বড় কালির মতো স্পট বিশ্লেষণ করে শনাক্ত করে।",
        "model_precision": "96.40%"
    },
    
    "RED POSTMAN": {
        "name_en": "Red Postman",
        "name_bn": "রেড পোস্টম্যান (ডাকপিয়ন)",
        "scientific_name": "Heliconius erato",
        "family_en": "Nymphalidae (Longwing butterflies)",
        "family_bn": "নিমফ্যালিডি (লম্বা ডানা)",
        "color_primary": "#DC2626",
        "wingspan": "55 – 80 mm",
        "lifespan_en": "Up to 6 – 9 months",
        "lifespan_bn": "৬ – ৯ মাস পর্যন্ত",
        "habitat_en": "Shaded forest edges, secondary rainforest corridors, and riverine banks of Central and South America (from Mexico to Argentina).",
        "habitat_bn": "মেক্সিকো থেকে আর্জেন্টিনা পর্যন্ত মধ্য ও দক্ষিণ আমেরিকার আর্দ্র বনাঞ্চল ও নদীর তীরবর্তী ছায়াঘেরা অঞ্চল।",
        "appearance_en": "Slender, elongated velvety black wings featuring an electrifying crimson-pink forewing band and a subtle yellow streak along the hindwing.",
        "appearance_bn": "লম্বাটে গড়নের ভেলভেট কালো ডানা, যাতে বিদ্যুৎ চমকানোর মতো উজ্জ্বল টকটকে লাল ব্যান্ড এবং পেছনের ডানায় সরু হলুদ রেখা থাকে।",
        "superpower_en": "Pollen Digestion & Müllerian Mimicry: One of the rare butterflies capable of digesting nutritious pollen grains (not just sipping nectar), extracting amino acids that grant it an extraordinary 9-month lifespan. Toxic to predators.",
        "superpower_bn": "পরাগ হজম ও দীর্ঘায়ু: সাধারণ প্রজাপতিরা শুধু মধু পান করে, কিন্তু রেড পোস্টম্যান ফুলের আস্ত পরাগরেণু হজম করে প্রোটিন তৈরি করতে পারে! ফলে এরা একটানা ৯ মাস পর্যন্ত সুস্থ ও তরুণ থাকে।",
        "ai_attention_en": "Grad-CAM gradients lock firmly onto the distinctive elongated crimson-red forewing stripe and the slender wing aspect ratio.",
        "ai_attention_bn": "এআই মডেলের হিটম্যাপ সামনের ডানার উজ্জ্বল লম্বাটে লাল স্ট্রাইপ এবং বিশেষ সিলুয়েটের ওপর সরাসরি লক করে।",
        "model_precision": "92.17%"
    },
    
    "SOUTHERN DOGFACE": {
        "name_en": "Southern Dogface",
        "name_bn": "সাউদার্ন ডগফেস (কুকুরমুখো)",
        "scientific_name": "Zerene cesonia",
        "family_en": "Pieridae (Whites and Yellows • Sulphurs)",
        "family_bn": "পিয়েরিডি (হলুদ ও সাদা)",
        "color_primary": "#CA8A04",
        "wingspan": "54 – 67 mm",
        "lifespan_en": "3 – 5 weeks",
        "lifespan_bn": "৩ – ৫ সপ্তাহ",
        "habitat_en": "Open dry fields, prairies, thorn scrub, hillsides, and desert washes from the Southern United States down through Central America to South America.",
        "habitat_bn": "মার্কিন যুক্তরাষ্ট্রের দক্ষিণাঞ্চল, মেক্সিকো ও মধ্য আমেরিকার শুষ্ক প্রান্তর, পাহাড়ের ঢাল ও ঘাসবন।",
        "appearance_en": "Vibrant canary-yellow wings where the broad black forewing border carves out an uncanny, unmistakable silhouette of a poodle dog's profile, centered with a black eye dot.",
        "appearance_bn": "উজ্জ্বল কাঁচা হলুদ ডানা, যার চারপাশের কালো সীমানাটি মিলে অদ্ভুতভাবে একটি 'পুডল কুকুরের মুখের' স্পষ্ট প্রতিকৃতি তৈরি করে—যার মধ্যে একটি কালো চোখের বিন্দুও থাকে!",
        "superpower_en": "Ultraviolet Poodle Mimicry: The 'poodle face' pattern reflects distinct UV light wavelengths visible to potential mates and disorienting to avian predators who mistake the pattern for a larger animal's gaze.",
        "superpower_bn": "আল্ট্রাভায়োলেট চোখের দৃষ্টি: ডানার কুকুরমুখো প্রতিকৃতিটি বিশেষ অতিবেগুনি (UV) আলো প্রতিফলন করে, যার ফলে দূর থেকে কোনো শিকারী পাখি একে কোনো বড় প্রাণীর চোখ মনে করে ভয় পায়!",
        "ai_attention_en": "Neural attention zeroes in on the sharp contrast between the canary-yellow disc and the black canine silhouette contour on the forewing apex.",
        "ai_attention_bn": "এআই মডেল হলুদ ডানার ওপর খোদাই করা অনন্য কুকুরমুখো কালো প্রতিকৃতি ও সেন্ট্রাল আই-ডট চিনে সহজেই প্রজাতি নিশ্চিত করে।",
        "model_precision": "89.10%"
    }
}
