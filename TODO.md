This is my roadmap for the post-LLM dev phase, once I have the server’s uptime stabilized. Since I’m migrating from **Portainer** to a **Manual/Dockge** setup and moving my drives from **USB to Internal**, I need to be careful with my file paths so my apps don’t start up "blank."

### Phase 0: My Docker Service Migration (The "Stack" Move)

#### 1. Standardizing My Directory Structure
On my new Ubuntu setup (or my current one before I swap the hardware), I’m going to create a central "home" for all my service configurations to make backups easier.
*   **My Plan:** Use `/opt/stacks` for my YAML files and `/opt/appdata` for the settings.
*   `sudo mkdir -p /opt/stacks /opt/appdata`

#### 2. The "Clean" Backup (On my old setup)
Before I move anything, I’ll stop my containers so no files are written while I’m copying them.
*   **Stop my services:** `docker stop plex qbittorrent sonarr radarr` (etc).
*   **Archive my configs:** I’ll create a compressed backup of my "config" folders (where my Plex database and Sonarr settings live).
    *   *Example:* `sudo tar -cvpzf appdata_backup.tar.gz /path/to/my/current/config/folders`

#### 3. Transfer and Extract (On my new setup)
*   I’ll move that `.tar.gz` to my new server (or my new OS install).
*   Then I'll extract it into my new standard location: `sudo tar -xvpzf appdata_backup.tar.gz -C /opt/appdata`

#### 4. Updating My Volume Paths in `compose.yaml`
This is where I need to be careful. Since I'm moving my drives from a USB enclosure to internal SATA, the mount points will change (e.g., from `/mnt/usb_drive` to `/mnt/data`).
*   I’ll open each of my `compose.yaml` files.
*   I need to update the **Left Side** of the colon in your volumes:
    ```yaml
    volumes:
      - /opt/appdata/plex:/config             # My new config path
      - /mnt/data/media/movies:/movies       # My new internal drive path
    ```

#### 5. Handling My Permissions (PUID/PGID Check)
I need to make sure my containers run as my specific user.
*   I’ll check my IDs by typing `id` in the terminal (I'm aiming for `1000`).
*   I’ll ensure my `compose.yaml` files include:
    ```yaml
    environment:
      - PUID=1000
      - PGID=1000
    ```
*   **Applying Permissions:** I’ll fix the ownership of my moved data so my containers can write to them:
    *   `sudo chown -R $USER:$USER /opt/appdata`
    *   `sudo chown -R $USER:$USER /mnt/data` (my internal drives)

#### 6. Updating My Hardware Mapping (QuickSync)
Since I’m moving to the 8700K (and eventually the 14700K), I need to ensure Plex can see my Integrated Graphics (iGPU).
*   In my Plex `compose.yaml`, I’ll make sure the device mapping is set:
    ```yaml
    devices:
      - /dev/dri:/dev/dri
    ```

#### 7. Launching via Dockge
Instead of just running `docker compose up`, I’m going to point **Dockge** to my `/opt/stacks` folder.
*   It should find my folders automatically.
*   I’ll click "Update" or "Deploy" in the UI.
*   **My immediate check:** I need to check the logs to ensure Plex doesn't have "Permission Denied" errors and that qBittorrent sees my files.

#### 8. My "Resume" Check (qBittorrent)
If my downloads show "Missing Files," it's because the path *inside* the container changed.
*   **My Fix:** Inside the qBit UI, I’ll select all torrents -> Right Click -> "Set Location" -> Point it to the new path so it can re-check the files.

---

### Phase 0.5: Immediate User & Security Setup
*I can do these steps anytime to secure my private files and prepare the server for secondary users.*

1.  **Create secondary user accounts:**
    *   `sudo adduser username`
2.  **Grant Sudo (if I trust them with administrative access):**
    *   `sudo usermod -aG sudo username`
3.  **Lock my home folder:**
    *   `chmod 700 /home/myusername` (This prevents anyone—even other sudo users—from browsing my files without explicitly using `sudo`).
4.  **Establish a shared model folder:**
    *   `sudo mkdir -p /srv/models`
    *   `sudo chmod 755 /srv/models` (This allows me and any secondary users to read the same GGUF files without duplicating them).

---

### Phase 1: Remote Access & Management
*Setting up my "Service" layer.*

1.  **Install Tailscale:** I’ll put this on the server and my other devices. This is my "backdoor" for SSH and web UIs without opening ports.
2.  **Setup Cloudflare Tunnel:** I’ll keep Overseerr public-facing for friends. I might also set up Cloudflare Zero Trust (Email PIN) for Sonarr/Radarr if I want web access without a VPN.
3.  **Install Dockge:** I'll move my current Docker Compose files into `/opt/stacks/[service-name]` and use Dockge to manage them through a clean Web UI.
4.  **Setup "Homepage" Dashboard:** I’ll create a simple `homepage` container so other users and friends have one clean URL to see all available services (Overseerr, OpenWebUI, etc.).

---

### Phase 2: My Hardware Migration (Internalizing Drives)
*Moving from the USB enclosure to my Rosewill L4500U case.*

1.  **The Physical Move:** I'll move my HDDs from the USB enclosure into the internal Rosewill bays.
2.  **Mounting:** I'll identify the drives using `lsblk -f` and update my `/etc/fstab` so they auto-mount to a permanent location (e.g., `/mnt/data`).
3.  **Power Management:** I'll install `hdparm` and set spin-down timers for my media drives to save power and reduce heat now that they aren't in a USB enclosure.

---

### Phase 3: My "Premier" Overhaul (Hardware Install)
*This covers the physical upgrade and the critical BIOS update.*

**Part A: The BIOS Update (Crucial for 13th/14th Gen)**
*Since I am buying a 14th Gen CPU, my motherboard **must** have a microcode update to prevent CPU degradation.*

1.  **Download:** On my Windows PC, I'll grab the latest BIOS from the motherboard manufacturer's support page.
2.  **Prepare USB:** I'll format a stick to FAT32 and copy the file over.
3.  **Flash (Method 1 - Flashback):** If my board has a "BIOS Flashback" button, I'll update it **before** I even install the CPU.
4.  **Flash (Method 2 - BIOS Menu):** If no button, I'll install the CPU/RAM, boot into the BIOS, and use the built-in flash utility.
5.  **Critical Setting:** Once I'm updated, I'll make sure to select "Intel Default Settings" in the BIOS power profile.

---

### My Prospective Purchases (TBD)

**The "God-Server" Core Upgrade:**
*   **My CPU Choice:** **Intel i7-14700K** (Recommended over the i9 to better suit my Peerless Assassin cooler and case thermals).
*   **My Motherboard:** **Z790 DDR4 Motherboard** (I need to ensure it is specifically the "D4" or "DDR4" version to reuse my current RAM).

**My "Sidecar" Option:**
*   **Current Gear:** I’ll test my **GTX 1060 6GB** in the second PCIe slot for `Continue` autocomplete tasks first.
*   **Future Gear:** I'm considering an **Intel Arc A380** for dedicated AV1 QuickSync and AI sidecar tasks.

**My Power Delivery Plan:**
*   My current 850W solution is solid, but I'll double-check if I need more robust cabling for additional PCIe power leads.
*   I plan to migrate this PSU to my gaming PC and retire my old 650W to improve the safety and longevity of my new parts.

**Storage Upgrade Pathway**
*   My primary plex media drive is nearing capacity. 
*   **For Capcity** I plan to buy 20TB Seagate recertified drives from https://serverpartdeals.com/products/seagate-exos-st20000nm002c-20tb-7-2k-rpm-sata-6gb-s-512e-3-5-recertified-hard-drive
*   **For Performance** I plan to buy a 4TB 2.5in SATA SSD to do caching (law and order)
*   MergerFS + snapRAID
