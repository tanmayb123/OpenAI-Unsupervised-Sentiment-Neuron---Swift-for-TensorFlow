cat Imports.swift Model.swift VisGen.swift > main.swift
swiftc main.swift
rm main.swift
mv main visgen
